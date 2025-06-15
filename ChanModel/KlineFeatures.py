"""
K线特征和AL Brooks市场分类特征提取模块

本模块实现了:
1. K线基础特征提取
2. AL Brooks市场分类特征提取
"""

import numpy as np
from typing import Dict, List, Any, Optional

from KLine.KLine_Unit import CKLine_Unit
from Common.CEnum import BI_DIR


def extract_kline_basic_features(klu: CKLine_Unit, previous_klus: List[CKLine_Unit]) -> Dict[str, float]:
    """
    提取K线基础特征
    
    Args:
        klu: 当前K线
        previous_klus: 前N根K线列表
        
    Returns:
        K线基础特征字典
    """
    features = {}
    
    # 安全检查
    if not previous_klus:
        return features
    
    # --- 单根K线特征 ---
    # 计算实体大小
    body_size = abs(klu.close - klu.open)
    range_size = klu.high - klu.low
    
    # 实体比例，实体占整个K线的百分比
    features["body_ratio"] = body_size / range_size if range_size > 0 else 0
    
    # K线方向
    features["is_bull"] = 1.0 if klu.close > klu.open else -1.0
    
    # 上下影线比例
    if klu.close >= klu.open:  # 阳线
        features["upper_shadow_ratio"] = (klu.high - klu.close) / range_size if range_size > 0 else 0
        features["lower_shadow_ratio"] = (klu.open - klu.low) / range_size if range_size > 0 else 0
    else:  # 阴线
        features["upper_shadow_ratio"] = (klu.high - klu.open) / range_size if range_size > 0 else 0
        features["lower_shadow_ratio"] = (klu.close - klu.low) / range_size if range_size > 0 else 0
    
    # K线强度（收盘价在整个K线范围中的相对位置）
    features["bar_strength"] = (klu.close - klu.low) / (klu.high - klu.low) if range_size > 0 else 0.5
    
    # 条形图闭合位置（上、中、下三分位）
    if features["bar_strength"] >= 0.67:
        features["bar_closing_position"] = 1.0  # 上分位
    elif features["bar_strength"] <= 0.33:
        features["bar_closing_position"] = -1.0  # 下分位
    else:
        features["bar_closing_position"] = 0.0  # 中分位
    
    # --- 相对形态特征 ---
    if len(previous_klus) > 0:
        prev_klu = previous_klus[0]  # 前一根K线
        
        # 跳空缺口
        features["gap_up"] = 1.0 if klu.low > prev_klu.high else 0.0
        features["gap_down"] = 1.0 if klu.high < prev_klu.low else 0.0
        
        # 外包关系与内包关系
        features["is_outside"] = 1.0 if klu.high > prev_klu.high and klu.low < prev_klu.low else 0.0
        features["is_inside"] = 1.0 if klu.high < prev_klu.high and klu.low > prev_klu.low else 0.0
        
        # 价格变化
        features["price_change"] = (klu.close - prev_klu.close) / prev_klu.close if prev_klu.close > 0 else 0.0
        features["range_change"] = (klu.high - klu.low) / (prev_klu.high - prev_klu.low) if (prev_klu.high - prev_klu.low) > 0 else 1.0
    
    # 高点创新高/低点创新低
    if len(previous_klus) >= 5:
        # 过去5根K线的高低点
        high_points = [k.high for k in previous_klus[:5]]
        low_points = [k.low for k in previous_klus[:5]]
        
        features["is_hh"] = 1.0 if klu.high > max(high_points) else 0.0  # Higher High
        features["is_ll"] = 1.0 if klu.low < min(low_points) else 0.0    # Lower Low
        features["is_hl"] = 1.0 if klu.high < min(high_points) else 0.0  # Higher Low
        features["is_lh"] = 1.0 if klu.low > max(low_points) else 0.0    # Lower High
    
    return features


def extract_brooks_market_features(klu: CKLine_Unit, previous_klus: List[CKLine_Unit], 
                                   context: Dict[str, Any], lookback: int = 50) -> Dict[str, float]:
    """
    提取AL Brooks市场分类特征
    
    Args:
        klu: 当前K线
        previous_klus: 前N根K线列表
        context: 上下文信息（笔、线段、中枢等）
        lookback: 回溯K线数量
        
    Returns:
        AL Brooks市场分类特征字典
    """
    features = {}
    
    # 安全检查
    if len(previous_klus) < lookback:
        return features
    
    # 使用实际可用的回溯数量
    actual_lookback = min(lookback, len(previous_klus))
    
    # 收集价格序列（包括当前K线）
    closes = [klu.close] + [k.close for k in previous_klus[:actual_lookback-1]]
    highs = [klu.high] + [k.high for k in previous_klus[:actual_lookback-1]]
    lows = [klu.low] + [k.low for k in previous_klus[:actual_lookback-1]]
    
    # --- 突破特征 (Breakout) ---
    # 确定最近的价格区间
    lookback_range = 20  # 使用过去20根K线来判断区间
    if len(previous_klus) >= lookback_range:
        range_high = max([k.high for k in previous_klus[:lookback_range]])
        range_low = min([k.low for k in previous_klus[:lookback_range]])
        range_size = range_high - range_low
        
        # 突破强度（价格超出前期区间的幅度）
        if klu.close > range_high:  # 向上突破
            features["breakout_strength"] = (klu.close - range_high) / range_high if range_high > 0 else 0.0
            features["breakout_direction"] = 1.0
        elif klu.close < range_low:  # 向下突破
            features["breakout_strength"] = (range_low - klu.close) / range_low if range_low > 0 else 0.0
            features["breakout_direction"] = -1.0
        else:  # 无突破
            features["breakout_strength"] = 0.0
            features["breakout_direction"] = 0.0
        
        # 是否在回测突破位置
        if 1 <= len(previous_klus) <= 5:  # 最近1-5根K线内有突破
            prev_breakout = None
            for i, pk in enumerate(previous_klus[:5]):
                if pk.close > range_high or pk.high > range_high:  # 向上突破
                    prev_breakout = (i, 1, range_high)
                    break
                elif pk.close < range_low or pk.low < range_low:  # 向下突破
                    prev_breakout = (i, -1, range_low)
                    break
            
            if prev_breakout:
                idx, direction, level = prev_breakout
                # 判断是否在回测
                if direction == 1 and klu.low <= level <= klu.high:  # 上破后回测
                    features["breakout_retest"] = 1.0
                elif direction == -1 and klu.low <= level <= klu.high:  # 下破后回测
                    features["breakout_retest"] = -1.0
                else:
                    features["breakout_retest"] = 0.0
            else:
                features["breakout_retest"] = 0.0
        
        # 失败突破识别（突破后迅速回落）
        failed_breakout = 0.0
        if len(previous_klus) >= 3:
            # 检查前3根K线是否有突破
            for i in range(3):
                if i >= len(previous_klus):
                    break
                pk = previous_klus[i]
                # 向上突破但随后回落
                if (pk.close > range_high or pk.high > range_high) and klu.close < range_high:
                    failed_breakout = 1.0
                    break
                # 向下突破但随后反弹
                elif (pk.close < range_low or pk.low < range_low) and klu.close > range_low:
                    failed_breakout = -1.0
                    break
        
        features["failed_breakout"] = failed_breakout
    
    # --- 区间特征 (Trading Range) ---
    # 计算最近的高低点确定区间
    if len(previous_klus) >= 20:
        recent_highs = [klu.high] + [k.high for k in previous_klus[:20]]
        recent_lows = [klu.low] + [k.low for k in previous_klus[:20]]
        
        highest = max(recent_highs)
        lowest = min(recent_lows)
        
        # 区间宽度
        range_width = (highest - lowest) / lowest if lowest > 0 else 0.0
        features["range_width"] = range_width
        
        # 区间中的位置
        if highest > lowest:
            position = (klu.close - lowest) / (highest - lowest)
            features["range_position"] = position
        else:
            features["range_position"] = 0.5
        
        # 区间测试次数（对上下边界的测试）
        upper_tests = 0
        lower_tests = 0
        
        for i in range(1, len(recent_highs)):
            # 如果接近上边界(5%以内)
            if recent_highs[i] > (highest - 0.05 * (highest - lowest)):
                upper_tests += 1
            # 如果接近下边界(5%以内)
            if recent_lows[i] < (lowest + 0.05 * (highest - lowest)):
                lower_tests += 1
        
        features["upper_bound_tests"] = min(upper_tests / 5.0, 1.0)  # 归一化
        features["lower_bound_tests"] = min(lower_tests / 5.0, 1.0)  # 归一化
        
        # 区间成熟度（基于测试次数和形成时间）
        features["range_maturity"] = min((upper_tests + lower_tests) / 10.0, 1.0)  # 归一化
    
    # --- 通道特征 (Channel) ---
    # 使用线性回归来拟合价格通道
    if len(closes) >= 20:
        x = np.arange(20)
        y_mid = np.array(closes[:20])
        y_high = np.array(highs[:20])
        y_low = np.array(lows[:20])
        
        # 中轨线性回归
        slope_mid, intercept_mid = np.polyfit(x, y_mid, 1)
        
        # 计算通道方向和斜率
        features["channel_direction"] = np.sign(slope_mid)
        features["channel_slope"] = slope_mid / np.mean(y_mid) if np.mean(y_mid) > 0 else 0.0
        
        # 计算通道宽度
        trend_line_mid = intercept_mid + slope_mid * x
        deviations = np.concatenate([
            y_high - trend_line_mid,  # 上偏差
            trend_line_mid - y_low    # 下偏差
        ])
        channel_width = np.mean(deviations) * 2
        features["channel_width"] = channel_width / np.mean(y_mid) if np.mean(y_mid) > 0 else 0.0
        
        # 通道质量（价格对通道线的吻合程度）
        residuals = y_mid - trend_line_mid
        channel_quality = 1.0 - min(np.std(residuals) / (channel_width/2), 1.0) if channel_width > 0 else 0.0
        features["channel_quality"] = channel_quality
        
        # 当前价格在通道中的位置
        current_trend_value = intercept_mid + slope_mid * 0  # x=0是当前位置
        if channel_width > 0:
            channel_position = (klu.close - (current_trend_value - channel_width/2)) / channel_width
            features["channel_position"] = max(0.0, min(1.0, channel_position))  # 归一化到[0,1]
        else:
            features["channel_position"] = 0.5
    
    # --- 通道宽窄特征 ---
    if "channel_width" in features:
        # 通道宽窄判断（使用相对历史标准差）
        recent_widths = []
        for i in range(5, min(30, len(previous_klus)), 5):
            if i+20 <= len(previous_klus):
                sub_highs = [k.high for k in previous_klus[i:i+20]]
                sub_lows = [k.low for k in previous_klus[i:i+20]]
                sub_close = [k.close for k in previous_klus[i:i+20]]
                
                x = np.arange(20)
                y_mid = np.array(sub_close)
                slope, intercept = np.polyfit(x, y_mid, 1)
                trend_line = intercept + slope * x
                
                deviations = np.concatenate([
                    np.array(sub_highs) - trend_line,
                    trend_line - np.array(sub_lows)
                ])
                sub_width = np.mean(deviations) * 2
                recent_widths.append(sub_width / np.mean(y_mid) if np.mean(y_mid) > 0 else 0.0)
        
        if recent_widths:
            avg_width = np.mean(recent_widths)
            # 通道宽窄分类
            if features["channel_width"] < 0.7 * avg_width:
                features["channel_type"] = -1.0  # 窄通道
            elif features["channel_width"] > 1.3 * avg_width:
                features["channel_type"] = 1.0   # 宽通道
            else:
                features["channel_type"] = 0.0   # 正常通道
            
            # 通道宽度变化
            features["channel_width_change"] = features["channel_width"] / avg_width - 1.0
        
        # 通道渗透（价格对通道边界的突破尝试）
        if "channel_position" in features:
            if features["channel_position"] > 0.95:
                features["channel_penetration"] = 1.0  # 接近上边界
            elif features["channel_position"] < 0.05:
                features["channel_penetration"] = -1.0  # 接近下边界
            else:
                features["channel_penetration"] = 0.0  # 在通道内部
    
    return features 