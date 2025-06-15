"""
价格水平特征提取模块

本模块实现重要价格水平识别和分析，包括：
1. 关键支撑阻力位识别
2. 价格水平测试次数统计
3. 价格突破强度计算
4. 与缠论结构整合的价格水平分析
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from KLine.KLine_Unit import CKLine_Unit
from Common.CEnum import BI_DIR
from Chan import CChan


def extract_price_level_features(klu: CKLine_Unit, previous_klus: List[CKLine_Unit],
                               chan_snapshot: CChan, context: Dict[str, Any]) -> Dict[str, float]:
    """
    提取价格水平特征
    
    Args:
        klu: 当前K线
        previous_klus: 前N根K线列表
        chan_snapshot: 缠论快照
        context: 上下文信息（笔、线段、中枢等）
        
    Returns:
        价格水平特征字典
    """
    features = {}
    
    # 安全检查
    if not previous_klus or len(previous_klus) < 20:
        return features
    
    # 收集价格序列
    closes = [klu.close] + [k.close for k in previous_klus]
    highs = [klu.high] + [k.high for k in previous_klus]
    lows = [klu.low] + [k.low for k in previous_klus]
    
    # --- 关键价格水平识别 ---
    levels_info = identify_key_levels(klu, previous_klus, lookback=100, min_touches=2)
    
    # 按距离排序，获取最近的支撑位和阻力位
    support_levels = [(level, info) for level, info in levels_info.items() if info['type'] == '支撑']
    resistance_levels = [(level, info) for level, info in levels_info.items() if info['type'] == '阻力']
    
    support_levels.sort(key=lambda x: x[1]['distance'])
    resistance_levels.sort(key=lambda x: x[1]['distance'])
    
    # 最近的支撑位距离和强度
    if support_levels:
        nearest_support = support_levels[0][1]
        features["nearest_support_distance"] = nearest_support['distance'] / 100.0  # 归一化到0-1
        features["nearest_support_strength"] = nearest_support['touches'] / 10.0  # 归一化
        features["key_support_price"] = nearest_support['price']
    
    # 最近的阻力位距离和强度
    if resistance_levels:
        nearest_resistance = resistance_levels[0][1]
        features["nearest_resistance_distance"] = nearest_resistance['distance'] / 100.0  # 归一化到0-1
        features["nearest_resistance_strength"] = nearest_resistance['touches'] / 10.0  # 归一化
        features["key_resistance_price"] = nearest_resistance['price']
    
    # 关键价格水平总数
    features["key_level_count"] = len(levels_info) / 20.0  # 归一化
    
    # 价格位于关键水平之间的相对位置
    if support_levels and resistance_levels:
        nearest_support_price = support_levels[0][1]['price']
        nearest_resistance_price = resistance_levels[0][1]['price']
        if nearest_resistance_price > nearest_support_price:
            level_range = nearest_resistance_price - nearest_support_price
            if level_range > 0:
                relative_position = (klu.close - nearest_support_price) / level_range
                features["price_level_position"] = relative_position
    
    # --- 支撑阻力测试次数 ---
    level_tests = calculate_key_level_tests(klu, previous_klus, levels_info, lookback=50)
    features.update({k: v / 10.0 for k, v in level_tests.items()})  # 归一化测试次数
    
    # --- 结合缠论结构的价格水平分析 ---
    # 中枢与支撑阻力位的关系
    if 'current_zs' in context and context['current_zs']:
        zs = context['current_zs']
        zs_high = zs.high
        zs_low = zs.low
        
        # 计算中枢与支撑阻力位的重叠度
        for level_name, level_info in levels_info.items():
            level_price = level_info['price']
            
            # 检查价格水平是否在中枢范围内
            if zs_low <= level_price <= zs_high:
                if level_info['type'] == '支撑':
                    features["zs_support_overlap"] = 1.0
                else:
                    features["zs_resistance_overlap"] = 1.0
    
    # 笔转向点形成的价格水平
    if 'bi_list' in context and len(context['bi_list']) >= 3:
        bi_reversal_levels = []
        recent_bis = context['bi_list'][-3:]
        
        for i in range(1, len(recent_bis)):
            prev_bi = recent_bis[i-1]
            curr_bi = recent_bis[i]
            
            if prev_bi.dir != curr_bi.dir:
                # 这是一个转向点
                reversal_price = prev_bi.get_end_val()
                bi_reversal_levels.append(reversal_price)
        
        # 检查当前价格是否接近笔转向点形成的水平
        for level_price in bi_reversal_levels:
            distance = abs(klu.close - level_price) / level_price if level_price > 0 else 0
            if distance < 0.01:  # 1%内视为接近
                features["bi_reversal_level_active"] = 1.0
                break
    
    # 线段转向点形成的价格水平
    if 'seg_list' in context and len(context['seg_list']) >= 3:
        seg_reversal_levels = []
        recent_segs = context['seg_list'][-3:]
        
        for i in range(1, len(recent_segs)):
            prev_seg = recent_segs[i-1]
            curr_seg = recent_segs[i]
            
            if prev_seg.dir != curr_seg.dir:
                # 这是一个转向点
                reversal_price = prev_seg.get_end_val()
                seg_reversal_levels.append(reversal_price)
        
        # 检查当前价格是否接近线段转向点形成的水平
        for level_price in seg_reversal_levels:
            distance = abs(klu.close - level_price) / level_price if level_price > 0 else 0
            if distance < 0.02:  # 2%内视为接近
                features["xd_reversal_level_active"] = 1.0
                features["xd_reversal_level_distance"] = distance * 50  # 归一化到0-1
                break
    
    # --- 突破分析 ---
    # 结合价格水平和走势的突破特征
    breakout_features = identify_breakout_features(klu, previous_klus, levels_info)
    features.update(breakout_features)
    
    return features


def identify_key_levels(klu: CKLine_Unit, previous_klus: List[CKLine_Unit], 
                        lookback: int = 100, min_touches: int = 2) -> Dict[str, Dict[str, Any]]:
    """
    识别关键价格水平
    
    Args:
        klu: 当前K线
        previous_klus: 前N根K线列表
        lookback: 回溯的K线数量
        min_touches: 被视为关键水平的最小触碰次数
        
    Returns:
        关键价格水平信息字典
    """
    # 限制回溯长度
    prev_klus = previous_klus[:min(lookback, len(previous_klus))]
    
    if not prev_klus:
        return {}
    
    # 提取历史价格数据
    highs = [k.high for k in prev_klus]
    lows = [k.low for k in prev_klus]
    
    # 识别潜在支撑阻力位的算法
    # 方法1：计算价格直方图，找出高密度区域
    price_range = max(highs) - min(lows)
    bin_width = price_range / 100  # 分100个区间
    
    # 创建价格区间
    bins = np.arange(min(lows), max(highs) + bin_width, bin_width)
    
    # 对高低点分别计算直方图
    high_hist, high_bins = np.histogram(highs, bins=bins)
    low_hist, _ = np.histogram(lows, bins=bins)
    
    # 合并高低点直方图
    combined_hist = high_hist + low_hist
    
    # 找出密度高的区域作为潜在支撑阻力位
    key_levels = []
    bin_centers = (high_bins[:-1] + high_bins[1:]) / 2
    
    for i, count in enumerate(combined_hist):
        if count >= min_touches:  # 至少被触及min_touches次
            key_levels.append((bin_centers[i], count))
    
    # 合并相近的水平
    merged_levels = []
    if key_levels:
        current_level = key_levels[0]
        
        for i in range(1, len(key_levels)):
            # 如果当前价格与前一个价格相差小于一定阈值，则合并
            if abs(key_levels[i][0] - current_level[0]) < bin_width * 2:
                # 取加权平均作为合并后的价格水平
                weight1 = current_level[1]
                weight2 = key_levels[i][1]
                weighted_price = (current_level[0] * weight1 + key_levels[i][0] * weight2) / (weight1 + weight2)
                current_level = (weighted_price, weight1 + weight2)
            else:
                merged_levels.append(current_level)
                current_level = key_levels[i]
        
        merged_levels.append(current_level)
    
    # 排序关键水平（价格从低到高）
    sorted_levels = sorted(merged_levels, key=lambda x: x[0])
    
    # 确定每个水平是支撑还是阻力
    levels_info = {}
    current_price = klu.close
    
    for level, count in sorted_levels:
        level_type = "支撑" if level < current_price else "阻力"
        distance = abs(current_price - level) / current_price * 100  # 距离当前价的百分比
        key = f"{level_type}_{level:.2f}"
        levels_info[key] = {
            "price": level,
            "touches": count,
            "distance": distance,
            "type": level_type
        }
    
    return levels_info


def calculate_key_level_tests(klu: CKLine_Unit, previous_klus: List[CKLine_Unit], 
                             key_levels_info: Dict[str, Dict[str, Any]], lookback: int = 50) -> Dict[str, float]:
    """
    计算近期对关键水平的测试次数
    
    Args:
        klu: 当前K线
        previous_klus: 前N根K线列表
        key_levels_info: 关键水平信息
        lookback: 回溯的K线数量
        
    Returns:
        测试次数特征字典
    """
    # 限制回溯长度
    prev_klus = previous_klus[:min(lookback, len(previous_klus))]
    
    if not prev_klus or not key_levels_info:
        return {}
    
    # 加上当前K线
    all_klus = [klu] + prev_klus
    
    # 遍历每个关键水平
    level_tests = {}
    threshold_pct = 0.001  # 价格接近阈值（0.1%）
    
    # 分别统计支撑位和阻力位的测试情况
    support_tests = 0
    resistance_tests = 0
    
    for level_name, level_data in key_levels_info.items():
        level_price = level_data["price"]
        level_type = level_data["type"]
        test_count = 0
        
        # 计算该水平的测试次数
        for i in range(len(all_klus)):
            k = all_klus[i]
            price_low = k.low
            price_high = k.high
            
            # 计算K线是否测试了该水平
            if (price_low <= level_price * (1 + threshold_pct) and
                price_high >= level_price * (1 - threshold_pct)):
                # 找到连续测试并只计一次
                if i == 0 or all_klus[i-1].low > level_price * (1 + threshold_pct) or all_klus[i-1].high < level_price * (1 - threshold_pct):
                    test_count += 1
                    
                    # 累加到支撑位或阻力位总测试次数
                    if level_type == "支撑":
                        support_tests += 1
                    else:
                        resistance_tests += 1
        
        # 存储该水平的测试次数
        level_tests[f"{level_name}_tests"] = test_count
    
    # 添加支撑位和阻力位的总测试次数
    level_tests["support_test_count"] = support_tests
    level_tests["resistance_test_count"] = resistance_tests
    
    return level_tests


def identify_breakout_features(klu: CKLine_Unit, previous_klus: List[CKLine_Unit], 
                             levels_info: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    识别价格突破特征
    
    Args:
        klu: 当前K线
        previous_klus: 前N根K线列表
        levels_info: 关键水平信息
        
    Returns:
        突破特征字典
    """
    features = {}
    
    if not previous_klus or len(previous_klus) < 5:
        return features
    
    # 前一根K线
    prev_klu = previous_klus[0]
    
    # 检查是否有突破关键水平
    for level_name, level_info in levels_info.items():
        level_price = level_info["price"]
        level_type = level_info["type"]
        
        # 检查突破支撑位
        if level_type == "支撑":
            if prev_klu.close > level_price and klu.close < level_price:
                features["support_breakout"] = 1.0
                features["breakout_strength"] = (level_price - klu.close) / level_price
                features["breakout_level_price"] = level_price
                break
        
        # 检查突破阻力位
        elif level_type == "阻力":
            if prev_klu.close < level_price and klu.close > level_price:
                features["resistance_breakout"] = 1.0
                features["breakout_strength"] = (klu.close - level_price) / level_price
                features["breakout_level_price"] = level_price
                break
    
    # 检查强势突破（大阳线/大阴线突破）
    body_size = abs(klu.close - klu.open)
    range_size = klu.high - klu.low
    body_ratio = body_size / range_size if range_size > 0 else 0.0
    
    if "support_breakout" in features or "resistance_breakout" in features:
        if body_ratio > 0.6:  # 大实体
            features["strong_breakout"] = 1.0
    
    # 检测突破后的确认情况
    if "breakout_level_price" in features:
        breakout_price = features["breakout_level_price"]
        is_upward = features.get("resistance_breakout", 0) > 0
        
        # 回测突破位的情况
        if (is_upward and klu.low <= breakout_price) or (not is_upward and klu.high >= breakout_price):
            features["breakout_retest"] = 1.0
    
    return features 