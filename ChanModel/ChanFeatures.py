"""
缠论结构特征提取模块

本模块实现了缠论相关的结构特征提取，包括：
1. 分型特征
2. 笔特征
3. 线段特征
4. 中枢特征
5. 买卖点特征
"""

from typing import Dict, List, Any, Optional

from KLine.KLine_Unit import CKLine_Unit
from Common.CEnum import BI_DIR, FX_TYPE, BSP_TYPE


def extract_chan_structure_features(klu: CKLine_Unit, context: Dict[str, Any]) -> Dict[str, float]:
    """
    提取缠论结构特征
    
    Args:
        klu: 当前K线
        context: 上下文信息（包含当前笔、线段、中枢等）
        
    Returns:
        缠论结构特征字典
    """
    features = {}
    
    # --- 分型特征 ---
    if klu.klc and hasattr(klu.klc, 'fx'):
        # 分型类型
        if klu.klc.fx == FX_TYPE.TOP:
            features["fx_type"] = 1.0  # 顶分型
        elif klu.klc.fx == FX_TYPE.BOTTOM:
            features["fx_type"] = -1.0  # 底分型
        else:
            features["fx_type"] = 0.0  # 无分型
        
        # 分型强度（如果是分型K线，计算其强度）
        if features["fx_type"] != 0:
            if features["fx_type"] == 1.0:  # 顶分型
                strength = (klu.klc.high - max(klu.klc.pre.high if klu.klc.pre else 0, 
                                              klu.klc.next.high if klu.klc.next else 0)) / klu.klc.high
            else:  # 底分型
                min_neighbor = min(klu.klc.pre.low if klu.klc.pre else float('inf'), 
                                  klu.klc.next.low if klu.klc.next else float('inf'))
                strength = (min_neighbor - klu.klc.low) / min_neighbor if min_neighbor > 0 else 0
            
            features["fx_strength"] = strength
    else:
        features["fx_type"] = 0.0
    
    # --- 笔特征 ---
    current_bi = context.get('current_bi')
    if current_bi:
        # 笔方向
        features["bi_direction"] = 1.0 if current_bi.dir == BI_DIR.UP else -1.0
        
        # 笔强度（起止点价格变化比例）
        begin_val = current_bi.get_begin_val()
        end_val = current_bi.get_end_val()
        if begin_val > 0:
            features["bi_strength"] = (end_val - begin_val) / begin_val
        else:
            features["bi_strength"] = 0.0
        
        # 将生成器转换为列表
        klc_list = list(current_bi.klc_lst)
        
        # 笔长度（包含的K线数量）
        features["bi_length"] = len(klc_list)
        
        # 笔斜率（价格变化/时间变化）
        # 由于时间单位可能不统一，这里简化为每单位K线的价格变化率
        if features["bi_length"] > 0:
            features["bi_slope"] = features["bi_strength"] / features["bi_length"]
        else:
            features["bi_slope"] = 0.0
        
        # 当前K线在笔中的位置
        for i, klc in enumerate(klc_list):
            if klu.klc == klc or klu in klc.lst:
                features["bi_position"] = i / len(klc_list) if len(klc_list) > 0 else 0.5
                break
        else:
            features["bi_position"] = 0.0
        
        # 笔确定性
        features["bi_is_sure"] = 1.0 if current_bi.is_sure else 0.0
        
        # 笔在其所属线段中的位置
        if current_bi.parent_seg and current_bi in current_bi.parent_seg.bi_list:
            bi_index = current_bi.parent_seg.bi_list.index(current_bi)
            features["bi_seg_position"] = bi_index / len(current_bi.parent_seg.bi_list) if len(current_bi.parent_seg.bi_list) > 0 else 0.5
    
    # --- 线段特征 ---
    current_seg = context.get('current_seg')
    if current_seg:
        # 线段方向
        features["seg_direction"] = 1.0 if current_seg.dir == BI_DIR.UP else -1.0
        
        # 线段强度（起止点价格变化比例）
        begin_val = current_seg.get_begin_val()
        end_val = current_seg.get_end_val()
        if begin_val > 0:
            features["seg_strength"] = (end_val - begin_val) / begin_val
        else:
            features["seg_strength"] = 0.0
        
        # 线段包含笔数
        features["seg_bi_count"] = len(current_seg.bi_list)
        
        # 线段确定性
        features["seg_is_sure"] = 1.0 if current_seg.is_sure else 0.0
        
        # 计算线段内部笔的特征序列指标
        if len(current_seg.bi_list) >= 3:
            # 计算特征序列的高低点
            feature_highs = []
            feature_lows = []
            
            for i in range(0, len(current_seg.bi_list) - 1, 2):
                if i + 1 < len(current_seg.bi_list):
                    if current_seg.dir == BI_DIR.UP:
                        # 上升线段中的下降笔高点
                        feature_highs.append(current_seg.bi_list[i].get_begin_val())
                        # 上升线段中的下降笔低点
                        feature_lows.append(current_seg.bi_list[i+1].get_end_val())
                    else:
                        # 下降线段中的上升笔低点
                        feature_lows.append(current_seg.bi_list[i].get_begin_val())
                        # 下降线段中的上升笔高点
                        feature_highs.append(current_seg.bi_list[i+1].get_end_val())
            
            # 特征序列指标
            if feature_highs and feature_lows:
                features["seg_eigen_ratio"] = (max(feature_highs) - min(feature_lows)) / begin_val if begin_val > 0 else 0.0
    
    # --- 中枢特征 ---
    current_zs = context.get('current_zs')
    if current_zs:
        # 中枢高低点
        features["zs_high"] = current_zs.high
        features["zs_low"] = current_zs.low
        
        # 中枢宽度
        if current_zs.low > 0:
            features["zs_width"] = (current_zs.high - current_zs.low) / current_zs.low
        else:
            features["zs_width"] = 0.0
        
        # 中枢包含笔数
        features["zs_bi_count"] = len(current_zs.bi_lst)
        
        # K线与中枢相对位置
        if klu.close > current_zs.high:
            features["klu_zs_position"] = 1.0  # 在中枢上方
        elif klu.close < current_zs.low:
            features["klu_zs_position"] = -1.0  # 在中枢下方
        else:
            features["klu_zs_position"] = 0.0  # 在中枢内
            # 在中枢内的相对位置
            features["klu_zs_relative_pos"] = (klu.close - current_zs.low) / (current_zs.high - current_zs.low) if (current_zs.high - current_zs.low) > 0 else 0.5
        
        # 中枢的峰值范围（最高点与最低点）
        if hasattr(current_zs, 'peak_high') and hasattr(current_zs, 'peak_low'):
            if current_zs.peak_low > 0:
                features["zs_peak_range"] = (current_zs.peak_high - current_zs.peak_low) / current_zs.peak_low
            else:
                features["zs_peak_range"] = 0.0
    
    # --- 买卖点特征 ---
    bsp_list = context.get('bsp_list', [])
    current_bsp = None
    
    # 查找当前K线的买卖点
    for bsp in bsp_list:
        if bsp.klu.idx == klu.idx:
            current_bsp = bsp
            break
    
    if current_bsp:
        features["is_bsp"] = 1.0
        features["is_buy"] = 1.0 if current_bsp.is_buy else 0.0
        
        # 买卖点类型
        bsp_type_value = 0.0
        for bs_type in current_bsp.type:
            # 将买卖点类型转换为数值特征
            type_map = {
                BSP_TYPE.T1: 1.0,
                BSP_TYPE.T2: 2.0,
                BSP_TYPE.T3A: 3.0,
                BSP_TYPE.T1P: 1.5,
                BSP_TYPE.T2S: 2.5,
                BSP_TYPE.T3B: 3.5,
            }
            if bs_type in type_map:
                # 取最小的类型值
                if bsp_type_value == 0.0 or type_map[bs_type] < bsp_type_value:
                    bsp_type_value = type_map[bs_type]
        
        features["bsp_type"] = bsp_type_value
        
        # 背驰程度（如果有）
        if hasattr(current_bsp, 'divergence_rate'):
            features["bsp_divergence"] = current_bsp.divergence_rate
    else:
        features["is_bsp"] = 0.0
        features["is_buy"] = 0.0
        features["bsp_type"] = 0.0
    
    return features 