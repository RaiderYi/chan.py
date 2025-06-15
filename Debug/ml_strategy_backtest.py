import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from typing import Dict, TypedDict, List
import xgboost as xgb
from matplotlib import pyplot as plt

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime

class TSampleInfo(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime

def strategy_feature(last_klu):
    """附加在买卖点上的额外特征示例"""
    return {
        "open_klu_rate": (last_klu.close - last_klu.open) / last_klu.open,
    }


def collect_samples(chan: CChan) -> Dict[int, TSampleInfo]:
    """运行策略收集买卖点特征"""
    bsp_dict: Dict[int, TSampleInfo] = {}
    for snapshot in chan.step_load():
        last_klu = snapshot[0][-1][-1]
        bsp_list = snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        cur_lv_chan = snapshot[0]
        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time,
            }
            bsp_dict[last_bsp.klu.idx]["feature"].add_feat(strategy_feature(last_klu))
    return bsp_dict


def gen_train_file(chan: CChan, bsp_dict: Dict[int, TSampleInfo], libsvm_path: str, meta_path: str) -> None:
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    feature_meta: Dict[str, int] = {}
    cur_idx = 0
    with open(libsvm_path, "w") as fid:
        for bsp_idx, info in bsp_dict.items():
            label = int(bsp_idx in bsp_academy)
            feats: List[str] = []
            for feat_name, value in info["feature"].items():
                if feat_name not in feature_meta:
                    feature_meta[feat_name] = cur_idx
                    cur_idx += 1
                feats.append(f"{feature_meta[feat_name]}:{value}")
            feats.sort(key=lambda x: int(x.split(":")[0]))
            fid.write(f"{label} {' '.join(feats)}\n")
    with open(meta_path, "w") as fid:
        fid.write(json.dumps(feature_meta))


def train_model(libsvm_path: str) -> xgb.Booster:
    dtrain = xgb.DMatrix(f"{libsvm_path}?format=libsvm")
    param = {
        "max_depth": 2,
        "eta": 0.3,
        "objective": "binary:logistic",
        "eval_metric": "auc",
    }
    model = xgb.train(param, dtrain=dtrain, num_boost_round=10)
    return model


def predict_bsp(model: xgb.Booster, last_bsp, meta: Dict[str, int]) -> float:
    missing = -9999999
    feat = [missing] * len(meta)
    for feat_name, value in last_bsp.features.items():
        if feat_name in meta:
            feat[meta[feat_name]] = value
    dtest = xgb.DMatrix([feat], missing=missing)
    return float(model.predict(dtest)[0])


def backtest(chan: CChan, model: xgb.Booster, meta: Dict[str, int], threshold: float = 0.6, hold_days: int = 5):
    cash = 100000.0
    position = 0.0
    entry_price = 0.0
    equity_curve = []
    klu_times = []
    day_cnt = 0
    for snapshot in chan.step_load():
        last_klu = snapshot[0][-1][-1]
        bsp_list = snapshot.get_bsp()
        if position:
            day_cnt += 1
            if day_cnt >= hold_days:
                cash = position * last_klu.close
                position = 0
        if bsp_list:
            last_bsp = bsp_list[-1]
            cur_lv_chan = snapshot[0]
            if cur_lv_chan[-2].idx == last_bsp.klu.klc.idx and not position:
                prob = predict_bsp(model, last_bsp, meta)
                if prob >= threshold and last_bsp.is_buy:
                    position = cash / last_klu.close
                    entry_price = last_klu.close
                    cash = 0
                    day_cnt = 0
        equity = cash + position * last_klu.close
        equity_curve.append(equity)
        klu_times.append(last_klu.time.to_str())
    plt.figure(figsize=(10, 4))
    plt.plot(klu_times, equity_curve)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    return equity_curve


def main():
    code = "sz.000001"
    begin_time = "2018-01-01"
    end_time = None
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig({"trigger_step": True})
    chan = CChan(code=code, begin_time=begin_time, end_time=end_time, data_src=data_src, lv_list=lv_list, config=config, autype=AUTYPE.QFQ)

    # 收集样本并训练模型
    bsp_dict = collect_samples(chan)
    libsvm_path = "feature.libsvm"
    meta_path = "feature.meta"
    gen_train_file(chan, bsp_dict, libsvm_path, meta_path)
    model = train_model(libsvm_path)
    json.dump(model.save_config(), open("model.json", "w"))

    # 重新初始化chan用于回测
    chan_bt = CChan(code=code, begin_time=begin_time, end_time=end_time, data_src=data_src, lv_list=lv_list, config=config, autype=AUTYPE.QFQ)
    meta = json.load(open(meta_path, "r"))
    backtest(chan_bt, model, meta)

if __name__ == "__main__":
    main()
