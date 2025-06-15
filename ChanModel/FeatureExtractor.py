"""
特征提取模块 - 融合缠论与AL Brooks价格行为学

本模块实现了缠论与AL Brooks价格行为学的完整特征体系，包括：
1. K线基础特征
2. AL Brooks市场分类特征
3. 缠论结构特征
4. 趋势与波动特征
5. 动量与能量特征
6. AL Brooks交易逻辑特征
7. 多时间框架特征
8. 价格水平特征
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import traceback

from Common.CEnum import BI_DIR
from Chan import CChan
from Bi.Bi import CBi
from Seg.Seg import CSeg
from ZS.ZS import CZS
from ChanConfig import CChanConfig
from BuySellPoint.BS_Point import CBS_Point
from KLine.KLine_Unit import CKLine_Unit
from KLine.KLine import CKLine
from ChanModel.Features import CFeatures

# 导入特征提取函数
from ChanModel.KlineFeatures import extract_kline_basic_features, extract_brooks_market_features
from ChanModel.ChanFeatures import extract_chan_structure_features
from ChanModel.TrendFeatures import extract_trend_features, extract_momentum_features
from ChanModel.LogicFeatures import extract_brooks_logic_features, extract_multi_timeframe_features
from ChanModel.PriceLevelFeatures import extract_price_level_features

class CFeatureExtractor:
    """
    特征提取器类：实现缠论与价格行为学的特征提取
    """
    
    def __init__(self, lookback_bars: int = 100, lookback_bis: int = 10, 
                 lookback_segs: int = 5, lookback_zs: int = 3):
        """
        初始化特征提取器
        
        Args:
            lookback_bars: 回溯K线数量
            lookback_bis: 回溯笔数量
            lookback_segs: 回溯线段数量
            lookback_zs: 回溯中枢数量
        """
        self.lookback_bars = lookback_bars
        self.lookback_bis = lookback_bis
        self.lookback_segs = lookback_segs
        self.lookback_zs = lookback_zs
    
    def extract_all_features(self, chan_snapshot: CChan, klu_idx: int = None) -> Dict[str, float]:
        """
        提取所有特征
        
        Args:
            chan_snapshot: 缠论快照
            klu_idx: 要分析的K线索引，默认为最后一根
            
        Returns:
            所有特征的字典
        """
        # 获取当前级别的数据
        data = chan_snapshot[0]
        
        # 获取目标K线
        if klu_idx is None:
            # 默认最后一根K线
            klu = data[-1][-1]
        else:
            # 查找指定索引的K线
            klu = None
            for klc in data:
                for k in klc.lst:
                    if k.idx == klu_idx:
                        klu = k
                        break
                if klu:
                    break
        
        if not klu:
            raise ValueError(f"找不到索引为 {klu_idx} 的K线")
        
        # 获取历史K线
        prev_klus = self._get_previous_klus(data, klu.idx)
        
        # 获取当前笔、线段、中枢
        context = self._get_context(chan_snapshot, klu, data)
        
        # 合并所有特征
        features = {}
        
        # 1. K线基础特征
        kl_features = extract_kline_basic_features(klu, prev_klus)
        features.update({f"kl_{k}": v for k, v in kl_features.items()})
        
        # 2. AL Brooks市场分类特征
        market_features = extract_brooks_market_features(klu, prev_klus, context)
        features.update({f"mk_{k}": v for k, v in market_features.items()})
        
        # 3. 缠论结构特征
        chan_features = extract_chan_structure_features(klu, context)
        features.update({f"ch_{k}": v for k, v in chan_features.items()})
        
        # 4. 趋势与波动特征
        trend_features = extract_trend_features(klu, prev_klus, context)
        features.update({f"tr_{k}": v for k, v in trend_features.items()})
        
        # 5. 动量与能量特征
        momentum_features = extract_momentum_features(klu, prev_klus)
        features.update({f"mo_{k}": v for k, v in momentum_features.items()})
        
        # 6. AL Brooks交易逻辑特征
        logic_features = extract_brooks_logic_features(klu, prev_klus, context)
        features.update({f"lg_{k}": v for k, v in logic_features.items()})
        
        # 7. 多时间框架特征
        tf_features = extract_multi_timeframe_features(chan_snapshot, klu)
        features.update({f"tf_{k}": v for k, v in tf_features.items()})
        
        # 8. 价格水平特征
        price_level_features = extract_price_level_features(klu, prev_klus, chan_snapshot, context)
        features.update({f"pl_{k}": v for k, v in price_level_features.items()})
        
        return features
    
    def _get_previous_klus(self, data, current_idx: int, max_count: int = None) -> List[CKLine_Unit]:
        """
        获取历史K线数据
        
        Args:
            data: 当前级别数据
            current_idx: 当前K线索引
            max_count: 最大回溯数量
            
        Returns:
            历史K线列表
        """
        max_count = max_count or self.lookback_bars
        prev_klus = []
        
        for klc in reversed(data.lst):
            for k in reversed(klc.lst):
                if k.idx < current_idx:
                    prev_klus.append(k)
                    if len(prev_klus) >= max_count:
                        return prev_klus
        
        return prev_klus
    
    def _get_context(self, chan_snapshot, klu, data):
        """
        获取当前K线的上下文信息（笔、线段、中枢等）
        
        Args:
            chan_snapshot: 缠论快照
            klu: 当前K线
            data: 当前级别数据
            
        Returns:
            上下文信息字典
        """
        context = {
            'current_bi': None,
            'current_seg': None,
            'current_zs': None,
            'bi_list': [],
            'seg_list': [],
            'zs_list': [],
            'bsp_list': []
        }
        
        # 获取当前笔
        for bi in data.bi_list:
            context['bi_list'].append(bi)
            if klu.klc in bi.klc_lst:
                context['current_bi'] = bi
        
        # 按时间顺序排序并限制数量
        context['bi_list'] = sorted(context['bi_list'], key=lambda x: x.idx)[-self.lookback_bis:]
        
        # 获取当前线段
        if context['current_bi']:
            for seg in data.seg_list.lst:
                context['seg_list'].append(seg)
                if context['current_bi'] in seg.bi_list:
                    context['current_seg'] = seg
        
        # 按时间顺序排序并限制数量
        context['seg_list'] = sorted(context['seg_list'], key=lambda x: x.idx)[-self.lookback_segs:]
        
        # 获取当前中枢
        for zs in data.zs_list.zs_lst:
            context['zs_list'].append(zs)
            if context['current_bi'] in zs.bi_lst:
                context['current_zs'] = zs
        
        # 按时间顺序排序并限制数量
        if context['zs_list']:
            # 中枢没有idx，使用第一个笔的idx来排序
            context['zs_list'] = sorted(context['zs_list'], key=lambda x: x.begin_bi.idx)[-self.lookback_zs:]
        
        # 获取买卖点列表
        context['bsp_list'] = chan_snapshot.get_bsp()
        
        return context
    
    def extract_base_features(self, chan, idx_or_position):
        """
        提取基础特征 (恢复到仅接受索引的版本)

        Args:
            chan: 缠论对象
            idx_or_position: 买卖点/K线索引 (注意：此版本仅正确处理整数索引)

        Returns:
            基础特征字典
        """
        features = {}
        klu = None
        bsp = None # 尝试获取 bsp 以提取类型等信息

        try:
            # 确认输入是整数索引
            if not isinstance(idx_or_position, int):
                print(f"警告 (extract_base_features): 期望整数索引，但收到 {type(idx_or_position)}。跳过特征提取。")
                return features
            klu_idx = idx_or_position

            # --- 确定 K 线类型和数据 ---
            kl_type = chan.cur_kl_type if hasattr(chan, 'cur_kl_type') else None
            if kl_type is None and hasattr(chan, 'kl_datas'):
                 kl_type = list(chan.kl_datas.keys())[0] if chan.kl_datas else None
            if kl_type is None:
                 print(f"警告(extract_base_features): 无法确定K线类型")
                 return features
            kl_data = chan.kl_datas.get(kl_type) if hasattr(chan, 'kl_datas') else None
            if kl_data is None or not hasattr(kl_data, 'lst'):
                 print(f"警告(extract_base_features): 未找到K线数据列表 {kl_type}")
                 return features

            # --- 通过索引获取 K 线 ---
            if 0 <= klu_idx < len(kl_data.lst):
                klu = kl_data.lst[klu_idx]
            else:
                print(f"警告(extract_base_features): 索引 {klu_idx} 超出 K 线列表范围 (0-{len(kl_data.lst)-1})")
                return features # 无法获取 K 线，直接返回

            # --- 尝试通过索引关联 BSP (如果存在 BSP 列表) ---
            if hasattr(kl_data, 'bs_point_lst') and hasattr(kl_data.bs_point_lst, 'lst'):
                 # 查找与 klu_idx 对应的 bsp (假设 klu.idx 匹配)
                 for point in kl_data.bs_point_lst.lst:
                    # 确保 point, klu, idx 都存在
                    if hasattr(point, 'klu') and point.klu and hasattr(point.klu, 'idx') and point.klu.idx == klu_idx:
                        bsp = point
                        break # 找到即停止

            # --- 提取特征 (基于获取到的 klu 和可选的 bsp) ---
            if klu:
                # 基本价格特征
                if self.use_price_features:
                     # 正确处理 CKLine 和 CKLine_Unit
                     if hasattr(klu, 'lst') and klu.lst: # CKLine (合并K线)
                         last_unit = klu.lst[-1]
                         features['price'] = last_unit.close if hasattr(last_unit, 'close') else 0.0
                         features['high'] = klu.high # 合并K线的最高价
                         features['low'] = klu.low   # 合并K线的最低价
                         features['vol'] = last_unit.vol if hasattr(last_unit, 'vol') else 0.0 # 最后一根的成交量? 或者合并的? 需确认逻辑
                     elif hasattr(klu, 'close'): # CKLine_Unit (单根K线)
                         features['price'] = klu.close
                         features['high'] = klu.high if hasattr(klu, 'high') else features['price']
                         features['low'] = klu.low if hasattr(klu, 'low') else features['price']
                         features['vol'] = klu.vol if hasattr(klu, 'vol') else 0.0
                     else: # 未知类型或缺少属性
                         features['price'] = 0.0
                         features['high'] = 0.0
                         features['low'] = 0.0
                         features['vol'] = 0.0

                     # 计算波动率
                     if features.get('high', 0) > 0:
                          features['volatility'] = (features.get('high', 0) - features.get('low', 0)) / features['high']
                     else:
                          features['volatility'] = 0.0

                # 指标特征
                if self.use_indicator_features:
                    if hasattr(klu, 'macd') and klu.macd:
                        # 假设指标对象属性为小写 (参考 enhanced_strategy_demo5.extract_features)
                        if hasattr(klu.macd, 'dif'): features['macd_dif'] = klu.macd.dif
                        if hasattr(klu.macd, 'dea'): features['macd_dea'] = klu.macd.dea
                        if hasattr(klu.macd, 'macd'): features['macd_macd'] = klu.macd.macd
                    if hasattr(klu, 'rsi') and klu.rsi is not None: # 检查非None
                        features['rsi'] = klu.rsi

            # 买卖点特征 (仅当 bsp 成功关联时提取)
            if bsp:
                features['is_buy'] = 1.0 if bsp.is_buy else 0.0
                if hasattr(bsp, 'type'):
                     type_list = bsp.type if isinstance(bsp.type, list) else [bsp.type]
                     for bsp_type_enum in type_list:
                         if bsp_type_enum: # 检查非空
                             type_name = bsp_type_enum.name if hasattr(bsp_type_enum, 'name') else str(bsp_type_enum)
                             type_str = type_name.replace("BSP_TYPE.", "") # 清理名称
                             features[f'type_{type_str}'] = 1.0 # 使用清理后的名称

        except Exception as e:
            print(f"提取基础特征时出错 for index {idx_or_position}: {e}")
            traceback.print_exc()

        return features
    
    # 各类特征提取函数将在接下来的模块中实现 

# 在模块级别添加特征缓存
_feature_cache = {}

def enhanced_feature_extraction(record, chan):
    """增强的特征提取方法，确保使用ChanModel/FeatureExtractor"""
    from ChanModel.FeatureExtractor import CFeatureExtractor
    
    # 初始化特征提取器
    feature_extractor = CFeatureExtractor()
    
    # 基本特征
    features = {
        "is_buy": 1.0 if record.is_buy else 0.0,
        "price": record.price if hasattr(record, "price") else 0.0
    }
    
    if hasattr(record, 'bsp') and record.bsp and hasattr(record.bsp, 'klu'):
        target_klu = record.bsp.klu
        
        # === 改进的KLU查找算法 ===
        current_kl_type = None
        if hasattr(target_klu, 'kl_type'):
            current_kl_type = target_klu.kl_type
        elif hasattr(record, 'kl_type') and record.kl_type:
            current_kl_type = record.kl_type
        elif chan.kl_datas:
            current_kl_type = list(chan.kl_datas.keys())[0]
            
        if current_kl_type and current_kl_type in chan.kl_datas:
            kl_list = chan.kl_datas[current_kl_type].lst
            
            # 尝试三种匹配方法
            # 1. 时间戳精确匹配
            # 2. 对象ID匹配 
            # 3. 价格和高低点匹配
            found_idx = None
            
            # 先检查target_klu是否直接在列表中
            try:
                direct_idx = kl_list.index(target_klu)
                found_idx = direct_idx
                print(f"找到精确匹配的KLU，索引为{direct_idx}")
            except (ValueError, TypeError):
                # 直接搜索失败，尝试时间戳匹配
                if hasattr(target_klu, 'time'):
                    target_time = target_klu.time
                    # 从列表末尾向前搜索提高效率
                    for i in range(len(kl_list)-1, max(0, len(kl_list)-200), -1):
                        klu = kl_list[i]
                        if (hasattr(klu, 'time') and 
                            klu.time.year == target_time.year and
                            klu.time.month == target_time.month and
                            klu.time.day == target_time.day and
                            klu.time.hour == target_time.hour and
                            klu.time.minute == target_time.minute):
                            found_idx = i
                            print(f"找到时间匹配的KLU，索引为{i}")
                            break
                
                # 如果时间匹配失败，尝试价格匹配
                if found_idx is None and hasattr(target_klu, 'high') and hasattr(target_klu, 'low'):
                    for i in range(len(kl_list)-1, max(0, len(kl_list)-200), -1):
                        klu = kl_list[i]
                        if (hasattr(klu, 'high') and hasattr(klu, 'low') and
                            abs(klu.high - target_klu.high) < 0.01 and
                            abs(klu.low - target_klu.low) < 0.01):
                            found_idx = i
                            print(f"找到价格匹配的KLU，索引为{i}")
                            break
            
            # 如果找到匹配的KLU，使用CFeatureExtractor提取特征
            if found_idx is not None:
                try:
                    print(f"使用CFeatureExtractor提取特征，索引为{found_idx}")
                    model_features = feature_extractor.extract_all_features(chan, found_idx)
                    features.update(model_features)
                    print(f"成功提取{len(model_features)}个ChanModel特征")
                    
                    # 在enhanced_feature_extraction函数中添加缓存逻辑
                    cache_key = f"{id(chan)}_{found_idx}"
                    if cache_key in _feature_cache:
                        return _feature_cache[cache_key]
                    
                    # 将提取的特征添加到缓存
                    _feature_cache[cache_key] = features
                    return features
                except Exception as e:
                    print(f"使用CFeatureExtractor提取特征失败: {e}")
    
    print("无法找到匹配的KLU或特征提取失败，使用基本特征")
    return features 

# 构建映射表
def build_klu_time_index(chan):
    time_index = {}
    for kl_type in chan.kl_datas:
        time_index[kl_type] = {}
        for i, klu in enumerate(chan.kl_datas[kl_type].lst):
            if hasattr(klu, 'time'):
                time_key = klu.time.to_str()
                time_index[kl_type][time_key] = i
    return time_index 