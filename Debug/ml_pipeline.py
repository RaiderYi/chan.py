import json
from typing import Dict, List

import xgboost as xgb

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.FeatureExtractor import CFeatureExtractor
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE


class SampleInfo:
    """Container for training sample information."""

    def __init__(self, feature: CFeatures, is_buy: bool, open_time):
        self.feature = feature
        self.is_buy = is_buy
        self.open_time = open_time


def collect_samples(code: str, begin_time: str, end_time: str) -> Dict[int, SampleInfo]:
    """Load data with CChan and collect BSP features."""

    config = CChanConfig({
        "trigger_step": True,
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": "1,2,3a,1p,2s,3b",
        "print_warning": False,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=DATA_SRC.CSV,
        lv_list=[KL_TYPE.K_DAY],
        config=config,
        autype=AUTYPE.QFQ,
    )

    feature_extractor = CFeatureExtractor()
    sample_dict: Dict[int, SampleInfo] = {}

    for snapshot in chan.step_load():
        last_klu = snapshot[0][-1][-1]
        bsp_list = snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        cur_lv_chan = snapshot[0]
        if last_bsp.klu.idx in sample_dict:
            continue
        if cur_lv_chan[-2].idx != last_bsp.klu.klc.idx:
            continue

        last_bsp.features.add_feat(feature_extractor.extract_all_features(snapshot, last_bsp.klu.idx))
        sample_dict[last_bsp.klu.idx] = SampleInfo(last_bsp.features, last_bsp.is_buy, last_klu.time)

    return sample_dict, chan


def save_training_data(samples: Dict[int, SampleInfo], chan: CChan) -> None:
    """Save features to libsvm format and meta mapping."""

    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    feature_meta: Dict[str, int] = {}
    cur_idx = 0

    with open("feature.libsvm", "w") as fid:
        for idx, info in samples.items():
            label = int(idx in bsp_academy)
            features = []
            for name, value in info.feature.items():
                if name not in feature_meta:
                    feature_meta[name] = cur_idx
                    cur_idx += 1
                features.append((feature_meta[name], value))
            features.sort(key=lambda x: x[0])
            feature_str = " ".join(f"{i}:{v}" for i, v in features)
            fid.write(f"{label} {feature_str}\n")

    with open("feature.meta", "w") as fid:
        json.dump(feature_meta, fid)


def train_model() -> xgb.Booster:
    """Train XGBoost model using saved libsvm file."""

    dtrain = xgb.DMatrix("feature.libsvm?format=libsvm")
    params = {
        "max_depth": 2,
        "eta": 0.3,
        "objective": "binary:logistic",
        "eval_metric": "auc",
    }
    model = xgb.train(params, dtrain=dtrain, num_boost_round=10)
    model.save_model("model.json")
    return model


def predict_realtime(chan: CChan, model: xgb.Booster, meta: Dict[str, int]) -> None:
    """Use trained model to score new BSP in real time."""

    feature_extractor = CFeatureExtractor()
    missing = -9999999
    treated_bsp_idx = set()

    for snapshot in chan.step_load():
        last_klu = snapshot[0][-1][-1]
        bsp_list = snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        cur_lv_chan = snapshot[0]
        if last_bsp.klu.idx in treated_bsp_idx:
            continue
        if cur_lv_chan[-2].idx != last_bsp.klu.klc.idx:
            continue

        last_bsp.features.add_feat(feature_extractor.extract_all_features(snapshot, last_bsp.klu.idx))
        feature_arr = [missing] * len(meta)
        for name, value in last_bsp.features.items():
            if name in meta:
                feature_arr[meta[name]] = value
        dtest = xgb.DMatrix([feature_arr], missing=missing)
        score = model.predict(dtest)[0]
        print(last_bsp.klu.time, score)
        treated_bsp_idx.add(last_bsp.klu.idx)


if __name__ == "__main__":
    CODE = "AAPL"
    BEGIN = "1985-01-01"
    END = "1990-12-31"

    samples, chan = collect_samples(CODE, BEGIN, END)
    save_training_data(samples, chan)
    bst = train_model()

    # Real-time evaluation on the same period for demonstration
    meta = json.load(open("feature.meta", "r"))
    chan_pred = CChan(
        code=CODE,
        begin_time=BEGIN,
        end_time=END,
        data_src=DATA_SRC.CSV,
        lv_list=[KL_TYPE.K_DAY],
        config=CChanConfig({"trigger_step": True}),
        autype=AUTYPE.QFQ,
    )
    predict_realtime(chan_pred, bst, meta)

