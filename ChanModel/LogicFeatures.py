"""
AL Brooks交易逻辑特征和多时间框架特征提取模块

本模块实现了:
1. AL Brooks交易逻辑特征提取
2. 多时间框架特征提取
3. 信号柱质量评估
4. 跟随柱强度评估
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from KLine.KLine_Unit import CKLine_Unit
from Common.CEnum import BI_DIR, KL_TYPE
from Chan import CChan


def extract_brooks_logic_features(klu: CKLine_Unit, previous_klus: List[CKLine_Unit], 
                                 context: Dict[str, Any]) -> Dict[str, float]:
    """
    提取AL Brooks交易逻辑特征
    
    Args:
        klu: 当前K线
        previous_klus: 前N根K线列表
        context: 上下文信息（笔、线段、中枢等）
        
    Returns:
        AL Brooks交易逻辑特征字典
    """
    features = {}
    
    # 安全检查
    if not previous_klus or len(previous_klus) < 10:
        return features
    
    # 收集价格序列
    closes = [klu.close] + [k.close for k in previous_klus]
    highs = [klu.high] + [k.high for k in previous_klus]
    lows = [klu.low] + [k.low for k in previous_klus]
    
    # --- 趋势后回调(Pullback)特征 ---
    # 识别趋势方向
    trend_direction = 0.0
    if len(closes) >= 20:
        # 使用简单的方法：比较当前价格和20周期前的价格
        if closes[0] > closes[19]:
            trend_direction = 1.0  # 上升趋势
        elif closes[0] < closes[19]:
            trend_direction = -1.0  # 下降趋势
    
    if trend_direction != 0:
        # 查找价格最近的极值点
        extreme_point = 0.0
        extreme_index = 0
        
        if trend_direction == 1.0:  # 上升趋势
            # 查找最近的高点
            for i in range(1, min(10, len(highs))):
                if i == 1 or highs[i] > extreme_point:
                    extreme_point = highs[i]
                    extreme_index = i
        else:  # 下降趋势
            # 查找最近的低点
            extreme_point = float('inf')
            for i in range(1, min(10, len(lows))):
                if i == 1 or lows[i] < extreme_point:
                    extreme_point = lows[i]
                    extreme_index = i
        
        # 计算回调深度
        if extreme_index > 0:
            if trend_direction == 1.0:  # 上升趋势回调
                pullback_depth = (extreme_point - klu.low) / (extreme_point - lows[extreme_index]) if (extreme_point - lows[extreme_index]) > 0 else 0.0
                features["pullback_depth"] = pullback_depth
                
                # 回调质量（回调过程中K线特征）
                strong_bars = 0
                weak_bars = 0
                for i in range(extreme_index):
                    if i >= len(previous_klus):
                        break
                    k = previous_klus[i]
                    if k.close < k.open:  # 阴线
                        body_ratio = (k.open - k.close) / (k.high - k.low) if (k.high - k.low) > 0 else 0.0
                        if body_ratio > 0.6:  # 大阴线
                            strong_bars += 1
                        elif body_ratio < 0.3:  # 小阴线
                            weak_bars += 1
                
                features["pullback_quality"] = (strong_bars * 2 + weak_bars) / (2 * extreme_index) if extreme_index > 0 else 0.0
                
            else:  # 下降趋势回调
                pullback_depth = (klu.high - extreme_point) / (highs[extreme_index] - extreme_point) if (highs[extreme_index] - extreme_point) > 0 else 0.0
                features["pullback_depth"] = pullback_depth
                
                # 回调质量
                strong_bars = 0
                weak_bars = 0
                for i in range(extreme_index):
                    if i >= len(previous_klus):
                        break
                    k = previous_klus[i]
                    if k.close > k.open:  # 阳线
                        body_ratio = (k.close - k.open) / (k.high - k.low) if (k.high - k.low) > 0 else 0.0
                        if body_ratio > 0.6:  # 大阳线
                            strong_bars += 1
                        elif body_ratio < 0.3:  # 小阳线
                            weak_bars += 1
                
                features["pullback_quality"] = (strong_bars * 2 + weak_bars) / (2 * extreme_index) if extreme_index > 0 else 0.0
        
        # --- 趋势恢复(With-trend Resumption)特征 ---
        if "pullback_depth" in features and features["pullback_depth"] > 0:
            # 连续同向K线作为恢复趋势的信号
            streak_count = 0
            for i in range(min(3, len(previous_klus))):
                if trend_direction == 1.0:  # 上升趋势
                    if previous_klus[i].close > previous_klus[i].open:  # 阳线
                        streak_count += 1
                    else:
                        break
                else:  # 下降趋势
                    if previous_klus[i].close < previous_klus[i].open:  # 阴线
                        streak_count += 1
                    else:
                        break
            
            # 恢复趋势的动力
            if streak_count > 0:
                momentum = 0.0
                for i in range(streak_count):
                    if i >= len(previous_klus):
                        break
                    k = previous_klus[i]
                    body_ratio = abs(k.close - k.open) / (k.high - k.low) if (k.high - k.low) > 0 else 0.0
                    momentum += body_ratio
                
                features["trend_resumption_strength"] = momentum / streak_count
                features["trend_resumption_confirmed"] = 1.0 if streak_count >= 2 else 0.0
    
    # --- 双重顶/底形态 ---
    if len(highs) >= 20 and len(lows) >= 20:
        # 查找近期的两个高点/低点
        high_points = []
        low_points = []
        
        # 寻找局部极值
        for i in range(1, min(19, len(highs)-1)):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                high_points.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                low_points.append((i, lows[i]))
        
        # 检查是否形成双顶/双底
        if len(high_points) >= 2:
            top1_idx, top1_val = high_points[0]
            top2_idx, top2_val = high_points[1]
            
            # 双顶条件：两个高点大致相等，中间有明显回落
            if abs(top1_val - top2_val) / top1_val < 0.03 and abs(top1_idx - top2_idx) >= 5:
                # 找中间的最低点
                mid_low = min(lows[min(top1_idx, top2_idx):max(top1_idx, top2_idx)+1])
                if (top1_val - mid_low) / top1_val > 0.03:  # 回落明显
                    features["double_top"] = 1.0
                    
                    # 颈线突破
                    if klu.close < mid_low:
                        features["neckline_break"] = -1.0  # 向下突破颈线
        
        if len(low_points) >= 2:
            bottom1_idx, bottom1_val = low_points[0]
            bottom2_idx, bottom2_val = low_points[1]
            
            # 双底条件：两个低点大致相等，中间有明显反弹
            if bottom1_val > 0 and abs(bottom1_val - bottom2_val) / bottom1_val < 0.03 and abs(bottom1_idx - bottom2_idx) >= 5:
                # 找中间的最高点
                mid_high = max(highs[min(bottom1_idx, bottom2_idx):max(bottom1_idx, bottom2_idx)+1])
                if (mid_high - bottom1_val) / bottom1_val > 0.03:  # 反弹明显
                    features["double_bottom"] = 1.0
                    
                    # 颈线突破
                    if klu.close > mid_high:
                        features["neckline_break"] = 1.0  # 向上突破颈线
    
    # --- 趋势线突破 ---
    # 简化版：使用高点连线和低点连线代替趋势线
    if len(highs) >= 20 and len(lows) >= 20:
        # 找出最近20根K线中的两个明显高点/低点
        high_points = []
        low_points = []
        
        for i in range(1, min(19, len(highs)-1)):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                high_points.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                low_points.append((i, lows[i]))
        
        # 如果有两个以上高点，计算下降趋势线
        if len(high_points) >= 2:
            # 使用最近的两个高点
            p1_idx, p1_val = high_points[0]
            p2_idx, p2_val = high_points[1]
            
            if p1_idx < p2_idx:  # 确保时间顺序
                p1_idx, p1_val, p2_idx, p2_val = p2_idx, p2_val, p1_idx, p1_val
            
            # 计算趋势线斜率
            slope = (p2_val - p1_val) / (p2_idx - p1_idx)
            
            # 延伸趋势线到当前K线
            trendline_val = p1_val - slope * p1_idx
            
            # 检查是否突破趋势线
            if slope < 0 and klu.high > trendline_val:
                features["trendline_break"] = 1.0  # 向上突破下降趋势线
                features["trendline_break_strength"] = (klu.high - trendline_val) / trendline_val if trendline_val > 0 else 0.0
        
        # 如果有两个以上低点，计算上升趋势线
        if len(low_points) >= 2:
            # 使用最近的两个低点
            p1_idx, p1_val = low_points[0]
            p2_idx, p2_val = low_points[1]
            
            if p1_idx < p2_idx:  # 确保时间顺序
                p1_idx, p1_val, p2_idx, p2_val = p2_idx, p2_val, p1_idx, p1_val
            
            # 计算趋势线斜率
            slope = (p2_val - p1_val) / (p2_idx - p1_idx)
            
            # 延伸趋势线到当前K线
            trendline_val = p1_val - slope * p1_idx
            
            # 检查是否突破趋势线
            if slope > 0 and klu.low < trendline_val:
                features["trendline_break"] = -1.0  # 向下突破上升趋势线
                features["trendline_break_strength"] = (trendline_val - klu.low) / trendline_val if trendline_val > 0 else 0.0
    
    # --- 交易信号强度 ---
    # 信号K线强度
    body_size = abs(klu.close - klu.open)
    range_size = klu.high - klu.low
    body_ratio = body_size / range_size if range_size > 0 else 0.0
    
    # 大实体K线
    features["signal_bar_strength"] = body_ratio
    
    # 增强的信号柱质量评估
    signal_bar_quality = calculate_signal_bar_quality(klu, previous_klus[:5])
    features.update(signal_bar_quality)
    
    # 跟随柱强度评估
    if len(previous_klus) >= 2:
        follow_through_features = calculate_follow_through_strength(
            previous_klus[0], klu, previous_klus[1:5]
        )
        features.update(follow_through_features)
    
    # K线位置（是否处于近期波动范围的极端位置）
    if len(highs) >= 10 and len(lows) >= 10:
        recent_high = max(highs[1:10])
        recent_low = min(lows[1:10])
        
        if recent_high > recent_low:
            position = (klu.close - recent_low) / (recent_high - recent_low)
            features["signal_bar_position"] = position
            
            # 极端位置的K线
            if position > 0.9:
                features["extreme_high_position"] = 1.0
            elif position < 0.1:
                features["extreme_low_position"] = 1.0
    
    # 设置K线质量（信号前的盘整或准备阶段）
    if len(previous_klus) >= 5:
        # 检查之前的K线是否形成紧密盘整
        highs_prev = [k.high for k in previous_klus[:5]]
        lows_prev = [k.low for k in previous_klus[:5]]
        
        high_range = max(highs_prev) - min(highs_prev)
        low_range = max(lows_prev) - min(lows_prev)
        
        avg_price = sum([k.close for k in previous_klus[:5]]) / 5
        
        # 盘整度：波动相对于均价的比例
        if avg_price > 0:
            consolidation = 1.0 - ((high_range + low_range) / 2) / avg_price
            features["setup_consolidation"] = max(0.0, min(consolidation, 1.0))
        
        # 方向一致性：连续的同向K线
        direction_count = 0
        current_direction = 1 if klu.close > klu.open else -1
        
        for i in range(min(3, len(previous_klus))):
            k = previous_klus[i]
            k_direction = 1 if k.close > k.open else -1
            
            if k_direction == current_direction:
                direction_count += 1
            else:
                break
        
        features["setup_direction_consistency"] = direction_count / 3.0
    
    # 后续确认K线
    if "is_bsp" in context.get('ch_features', {}) and context['ch_features']['is_bsp'] > 0:
        # 如果当前K线是买卖点，检查前面K线的确认情况
        is_buy = context['ch_features'].get('is_buy', 0) > 0
        
        confirmation_strength = 0.0
        for i in range(min(3, len(previous_klus))):
            k = previous_klus[i]
            
            if is_buy:  # 买点
                if k.close > k.open:  # 阳线，支持买点
                    confirmation_strength += body_ratio * 0.3
                else:  # 阴线，减弱买点
                    confirmation_strength -= body_ratio * 0.3
            else:  # 卖点
                if k.close < k.open:  # 阴线，支持卖点
                    confirmation_strength += body_ratio * 0.3
                else:  # 阳线，减弱卖点
                    confirmation_strength -= body_ratio * 0.3
        
        features["confirmation_strength"] = max(-1.0, min(confirmation_strength, 1.0))
    
    # 失败信号特征
    # 使用缠论框架计算的买卖点进行判断
    if "is_bsp" in context.get('ch_features', {}) and context['ch_features']['is_bsp'] > 0:
        # 当前K线是买卖点
        is_buy = context['ch_features'].get('is_buy', 0) > 0
        
        if is_buy:  # 买点
            # 买点后价格继续下跌是失败信号
            for i in range(min(5, len(previous_klus))):
                if i >= len(previous_klus):
                    break
                if previous_klus[i].low < klu.low:
                    features["failed_signal"] = 1.0
                    break
        else:  # 卖点
            # 卖点后价格继续上涨是失败信号
            for i in range(min(5, len(previous_klus))):
                if i >= len(previous_klus):
                    break
                if previous_klus[i].high > klu.high:
                    features["failed_signal"] = 1.0
                    break
    
    return features


def calculate_signal_bar_quality(klu: CKLine_Unit, previous_klus: List[CKLine_Unit]) -> Dict[str, float]:
    """
    评估信号柱质量
    
    Args:
        klu: 当前K线（信号柱）
        previous_klus: 前N根K线列表
        
    Returns:
        信号柱质量特征字典
    """
    features = {}
    
    # 安全检查
    if not previous_klus:
        return features
    
    # 1. 计算实体大小得分
    body_size = abs(klu.close - klu.open)
    range_size = klu.high - klu.low
    body_ratio = body_size / range_size if range_size > 0 else 0.0
    
    # 实体得分
    if body_ratio >= 0.7:  # 大实体
        body_score = 1.0
    elif body_ratio >= 0.5:  # 中等实体
        body_score = 0.7
    elif body_ratio >= 0.3:  # 小实体
        body_score = 0.4
    else:  # 十字星
        body_score = 0.1
    
    features["signal_bar_body_score"] = body_score
    
    # 2. 计算方向清晰度得分
    is_bull = klu.close > klu.open
    is_bear = klu.close < klu.open
    
    # 如果是看涨信号，阳线质量更高；如果是看跌信号，阴线质量更高
    if is_bull:
        # 计算上影线比例
        upper_shadow = klu.high - max(klu.open, klu.close)
        upper_shadow_ratio = upper_shadow / range_size if range_size > 0 else 0.0
        
        # 上影线小得分高
        if upper_shadow_ratio <= 0.1:
            direction_score = 1.0
        elif upper_shadow_ratio <= 0.2:
            direction_score = 0.8
        elif upper_shadow_ratio <= 0.3:
            direction_score = 0.6
        else:
            direction_score = 0.3
    elif is_bear:
        # 计算下影线比例
        lower_shadow = min(klu.open, klu.close) - klu.low
        lower_shadow_ratio = lower_shadow / range_size if range_size > 0 else 0.0
        
        # 下影线小得分高
        if lower_shadow_ratio <= 0.1:
            direction_score = 1.0
        elif lower_shadow_ratio <= 0.2:
            direction_score = 0.8
        elif lower_shadow_ratio <= 0.3:
            direction_score = 0.6
        else:
            direction_score = 0.3
    else:  # 十字星
        direction_score = 0.1
    
    features["signal_bar_direction_score"] = direction_score
    
    # 3. 计算影线比例得分
    upper_shadow = klu.high - max(klu.open, klu.close)
    lower_shadow = min(klu.open, klu.close) - klu.low
    
    upper_shadow_ratio = upper_shadow / range_size if range_size > 0 else 0.0
    lower_shadow_ratio = lower_shadow / range_size if range_size > 0 else 0.0
    
    # 根据Al Brooks理论，无影线或单侧小影线的信号柱质量更高
    if upper_shadow_ratio <= 0.1 and lower_shadow_ratio <= 0.1:  # 无影线
        shadow_score = 1.0
    elif (is_bull and upper_shadow_ratio <= 0.1) or (is_bear and lower_shadow_ratio <= 0.1):  # 单侧无影线
        shadow_score = 0.9
    elif (is_bull and lower_shadow_ratio <= 0.2) or (is_bear and upper_shadow_ratio <= 0.2):  # 方向侧影线适中
        shadow_score = 0.7
    elif upper_shadow_ratio + lower_shadow_ratio <= 0.4:  # 总影线占比较小
        shadow_score = 0.5
    else:  # 影线占比大
        shadow_score = 0.3
    
    features["signal_bar_shadow_score"] = shadow_score
    
    # 4. 计算位置得分
    if len(previous_klus) >= 3:
        # 收集前几根K线的价格数据
        prev_highs = [k.high for k in previous_klus[:3]]
        prev_lows = [k.low for k in previous_klus[:3]]
        
        # 计算近期高低点
        recent_high = max(prev_highs)
        recent_low = min(prev_lows)
        price_range = recent_high - recent_low
        
        if price_range > 0:
            # 计算当前K线在近期波动区间的位置（0=低点，1=高点）
            if is_bull:
                # 看涨信号柱靠近低点质量更高
                position_ratio = (klu.close - recent_low) / price_range
                position_score = 1.0 - position_ratio  # 越靠近低点越好
            else:
                # 看跌信号柱靠近高点质量更高
                position_ratio = (klu.close - recent_low) / price_range
                position_score = position_ratio  # 越靠近高点越好
            
            # 归一化得分
            position_score = max(0.1, min(1.0, position_score))
        else:
            position_score = 0.5  # 默认中等位置
    else:
        position_score = 0.5  # 默认中等位置
    
    features["signal_bar_position_score"] = position_score
    
    # 5. 计算总体信号柱质量得分（加权平均）
    weights = {
        "body": 0.35,
        "direction": 0.25,
        "shadow": 0.25,
        "position": 0.15
    }
    
    signal_quality = (
        weights["body"] * body_score +
        weights["direction"] * direction_score +
        weights["shadow"] * shadow_score +
        weights["position"] * position_score
    )
    
    features["signal_bar_quality"] = signal_quality
    
    # 6. 测量信号柱相对前几根K线的大小比例
    if len(previous_klus) >= 3:
        prev_ranges = [k.high - k.low for k in previous_klus[:3]]
        avg_prev_range = sum(prev_ranges) / len(prev_ranges) if prev_ranges else 1.0
        
        # 信号柱大小相对于前几根K线平均大小的比例
        relative_size = range_size / avg_prev_range if avg_prev_range > 0 else 1.0
        features["signal_bar_relative_size"] = min(relative_size, 3.0) / 3.0  # 归一化到0-1
    
    return features


def calculate_follow_through_strength(
    follow_bar: CKLine_Unit, signal_bar: CKLine_Unit, previous_klus: List[CKLine_Unit]
) -> Dict[str, float]:
    """
    评估跟随柱强度
    
    Args:
        follow_bar: 跟随柱（当前K线的前一根）
        signal_bar: 信号柱（当前K线的前两根）
        previous_klus: 前N根K线列表
        
    Returns:
        跟随柱强度特征字典
    """
    features = {}
    
    # 安全检查
    if not follow_bar or not signal_bar:
        return features
    
    # 确定信号柱方向
    signal_is_bull = signal_bar.close > signal_bar.open
    signal_is_bear = signal_bar.close < signal_bar.open
    
    # 确定跟随柱方向
    follow_is_bull = follow_bar.close > follow_bar.open
    follow_is_bear = follow_bar.close < follow_bar.open
    
    # 1. 方向一致性得分
    if (signal_is_bull and follow_is_bull) or (signal_is_bear and follow_is_bear):
        direction_consistency = 1.0  # 方向一致
    else:
        direction_consistency = 0.0  # 方向相反
    
    features["follow_through_direction_consistency"] = direction_consistency
    
    # 2. 价格突破情况
    if signal_is_bull:
        # 看涨信号，检查跟随柱是否突破信号柱高点
        if follow_bar.high > signal_bar.high:
            features["follow_through_breakthrough"] = 1.0
            # 计算突破的幅度
            breakthrough_size = (follow_bar.high - signal_bar.high) / signal_bar.high if signal_bar.high > 0 else 0.0
            features["follow_through_breakthrough_size"] = min(breakthrough_size * 10, 1.0)  # 归一化到0-1
    else:
        # 看跌信号，检查跟随柱是否突破信号柱低点
        if follow_bar.low < signal_bar.low:
            features["follow_through_breakthrough"] = 1.0
            # 计算突破的幅度
            breakthrough_size = (signal_bar.low - follow_bar.low) / signal_bar.low if signal_bar.low > 0 else 0.0
            features["follow_through_breakthrough_size"] = min(breakthrough_size * 10, 1.0)  # 归一化到0-1
    
    # 3. 跟随柱强度
    follow_body_size = abs(follow_bar.close - follow_bar.open)
    follow_range_size = follow_bar.high - follow_bar.low
    follow_body_ratio = follow_body_size / follow_range_size if follow_range_size > 0 else 0.0
    
    features["follow_through_body_ratio"] = follow_body_ratio
    
    # 4. 跟随柱与信号柱的相对大小
    signal_range = signal_bar.high - signal_bar.low
    relative_size = follow_range_size / signal_range if signal_range > 0 else 1.0
    features["follow_through_relative_size"] = min(relative_size, 3.0) / 3.0  # 归一化到0-1
    
    # 5. 成交量关联性
    if hasattr(follow_bar, 'volume') and hasattr(signal_bar, 'volume'):
        if signal_bar.volume > 0:
            volume_ratio = follow_bar.volume / signal_bar.volume
            # 归一化到0-1，超过2倍算作1
            features["follow_through_volume_ratio"] = min(volume_ratio, 2.0) / 2.0
            
            # 方向性确认与成交量增加的组合得分
            if direction_consistency > 0 and follow_bar.volume > signal_bar.volume:
                features["follow_through_volume_confirmation"] = 1.0
            else:
                features["follow_through_volume_confirmation"] = 0.0
    
    # 6. 总体跟随柱强度得分
    weights = {
        "direction": 0.4,
        "body_ratio": 0.3,
        "breakthrough": 0.3
    }
    
    # 基础得分计算
    base_score = (
        weights["direction"] * direction_consistency +
        weights["body_ratio"] * follow_body_ratio
    )
    
    # 如果有突破，增加突破分数
    breakthrough_score = weights["breakthrough"] * features.get("follow_through_breakthrough_size", 0.0)
    if "follow_through_breakthrough" in features:
        total_score = base_score + breakthrough_score
    else:
        total_score = base_score
    
    features["follow_through_strength"] = total_score
    
    return features


def extract_multi_timeframe_features(chan_snapshot: CChan, klu: CKLine_Unit) -> Dict[str, float]:
    """
    提取多时间框架特征
    
    Args:
        chan_snapshot: 缠论快照
        klu: 当前K线
        
    Returns:
        多时间框架特征字典
    """
    features = {}
    
    # 获取所有可用级别
    available_levels = list(chan_snapshot.kl_datas.keys())
    
    # 级别数量检查
    if len(available_levels) <= 1:
        return features
    
    # 当前级别对应的CChan对象
    current_level_data = None
    current_level = None
    
    # 查找当前K线所在级别
    for level, data in chan_snapshot.kl_datas.items():
        for klc in data.lst:
            for k in klc.lst:
                if k.idx == klu.idx:
                    current_level_data = data
                    current_level = level
                    break
            if current_level:
                break
        if current_level:
            break
    
    if not current_level:
        return features
    
    # 查找上一级别
    level_order = [KL_TYPE.K_1M, KL_TYPE.K_5M, KL_TYPE.K_15M, KL_TYPE.K_30M, 
                   KL_TYPE.K_60M, KL_TYPE.K_DAY, KL_TYPE.K_WEEK, KL_TYPE.K_MONTH]
    
    current_idx = -1
    for idx, level in enumerate(level_order):
        if level == current_level:
            current_idx = idx
            break
    
    if current_idx == -1 or current_idx >= len(level_order) - 1:
        # 当前已经是最高级别或无法确定级别顺序
        return features
    
    upper_level = level_order[current_idx + 1]
    
    # 检查上一级别是否存在
    if upper_level not in chan_snapshot.kl_datas:
        return features
    
    upper_level_data = chan_snapshot.kl_datas[upper_level]
    
    # --- 上级别确认 ---
    # 根据当前K线的父级别K线来获取上下文
    sup_kl = klu.sup_kl
    
    if sup_kl:
        # 在父级别中寻找相应的缠论元素
        
        # 在父级别中查找对应的买卖点
        upper_bsp_list = chan_snapshot.get_bsp(upper_level)
        upper_bsp_match = False
        
        for bsp in upper_bsp_list:
            if bsp.klu.idx == sup_kl.idx:
                upper_bsp_match = True
                features["upper_bsp_confirmation"] = 1.0
                features["upper_bsp_is_buy"] = 1.0 if bsp.is_buy else 0.0
                
                # 买卖点类型
                bsp_type_value = 0.0
                for bs_type in bsp.type:
                    # 将买卖点类型转换为数值特征
                    type_map = {
                        1: 1.0,  # BS1
                        2: 2.0,  # BS2
                        3: 3.0,  # BS3
                        4: 1.5,  # BS1_PEAK
                        5: 2.5,  # BS2_STRICT
                        6: 3.5,  # BS3_BEND
                    }
                    if bs_type in type_map:
                        # 取最小的类型值
                        if bsp_type_value == 0.0 or type_map[bs_type] < bsp_type_value:
                            bsp_type_value = type_map[bs_type]
                
                features["upper_bsp_type"] = bsp_type_value
                break
        
        if not upper_bsp_match:
            features["upper_bsp_confirmation"] = 0.0
        
        # 检查父级别的趋势方向
        upper_bi_list = []
        for bi in upper_level_data.bi_list:
            if sup_kl.klc in bi.klc_lst:
                upper_bi_list.append(bi)
        
        if upper_bi_list:
            current_upper_bi = upper_bi_list[-1]
            features["upper_bi_direction"] = 1.0 if current_upper_bi.dir == BI_DIR.UP else -1.0
            features["upper_bi_is_sure"] = 1.0 if current_upper_bi.is_sure else 0.0
        
        # 检查父级别的线段方向
        for seg in upper_level_data.seg_list.lst:
            for bi in upper_bi_list:
                if bi in seg.bi_list:
                    features["upper_seg_direction"] = 1.0 if seg.dir == BI_DIR.UP else -1.0
                    features["upper_seg_is_sure"] = 1.0 if seg.is_sure else 0.0
                    break
    
    # --- 下级别细分 ---
    # 当前K线包含的所有次级别K线
    if hasattr(klu, 'sub_kl_list') and klu.sub_kl_list:
        sub_kl_list = klu.sub_kl_list
        
        # 次级别K线数量
        features["sub_kl_count"] = len(sub_kl_list)
        
        # 次级别K线的方向一致性
        if len(sub_kl_list) > 0:
            up_count = sum(1 for k in sub_kl_list if k.close > k.open)
            down_count = sum(1 for k in sub_kl_list if k.close < k.open)
            
            if up_count > down_count:
                features["sub_kl_direction"] = 1.0  # 多上涨K线
                features["sub_kl_consensus"] = up_count / len(sub_kl_list)
            else:
                features["sub_kl_direction"] = -1.0  # 多下跌K线
                features["sub_kl_consensus"] = down_count / len(sub_kl_list)
        
        # 次级别开盘到收盘的路径特征
        if len(sub_kl_list) >= 3:
            # 计算开盘、最高、最低、收盘点的相对位置
            first_kl = sub_kl_list[0]
            last_kl = sub_kl_list[-1]
            
            max_high = max(k.high for k in sub_kl_list)
            min_low = min(k.low for k in sub_kl_list)
            
            high_idx = next((i for i, k in enumerate(sub_kl_list) if k.high == max_high), -1)
            low_idx = next((i for i, k in enumerate(sub_kl_list) if k.low == min_low), -1)
            
            # 标准化索引到[0,1]区间
            high_pos = high_idx / (len(sub_kl_list) - 1) if len(sub_kl_list) > 1 else 0.5
            low_pos = low_idx / (len(sub_kl_list) - 1) if len(sub_kl_list) > 1 else 0.5
            
            features["sub_kl_high_position"] = high_pos
            features["sub_kl_low_position"] = low_pos
            
            # 计算价格路径的方向性
            price_path = []
            for k in sub_kl_list:
                price_path.append(k.close)
            
            # 使用简单线性回归计算方向性
            x = np.arange(len(price_path))
            if len(price_path) > 1:
                slope, _ = np.polyfit(x, price_path, 1)
                features["sub_kl_path_direction"] = np.sign(slope)
                features["sub_kl_path_slope"] = slope / price_path[0] if price_path[0] > 0 else 0.0
    
    # --- 级别共振 ---
    # 计算当前级别和上级别的信号一致性
    if "upper_bsp_confirmation" in features and features["upper_bsp_confirmation"] > 0:
        # 查找当前K线的买卖点信息
        current_bsp_list = chan_snapshot.get_bsp(current_level)
        current_bsp = None
        
        for bsp in current_bsp_list:
            if bsp.klu.idx == klu.idx:
                current_bsp = bsp
                break
        
        if current_bsp:
            # 检查买卖方向是否一致
            upper_is_buy = features.get("upper_bsp_is_buy", 0) > 0
            current_is_buy = current_bsp.is_buy
            
            if upper_is_buy == current_is_buy:
                features["level_resonance"] = 1.0  # 完全共振
            else:
                features["level_resonance"] = -1.0  # 方向相反
        else:
            features["level_resonance"] = 0.0  # 当前级别无买卖点
    
    # --- 背驰与转折点 ---
    # 当前级别的背驰特征
    current_bi_list = []
    for bi in current_level_data.bi_list:
        if klu.klc in bi.klc_lst:
            current_bi_list.append(bi)
    
    if current_bi_list:
        current_bi = current_bi_list[-1]
        
        # 检查该笔是否有背驰标记
        if hasattr(current_bi, 'divergence') and current_bi.divergence:
            features["bi_divergence"] = 1.0
            
            # 背驰强度
            if hasattr(current_bi, 'divergence_rate'):
                features["bi_divergence_strength"] = current_bi.divergence_rate
        
        # 查找次级别确认信号
        if hasattr(klu, 'sub_kl_list') and klu.sub_kl_list:
            sub_kl_list = klu.sub_kl_list
            
            # 次级别的买卖点数量
            sub_bsp_count = 0
            sub_buy_count = 0
            sub_sell_count = 0
            
            for sub_kl in sub_kl_list:
                for level, data in chan_snapshot.kl_datas.items():
                    for bsp in chan_snapshot.get_bsp(level):
                        if bsp.klu.idx == sub_kl.idx:
                            sub_bsp_count += 1
                            if bsp.is_buy:
                                sub_buy_count += 1
                            else:
                                sub_sell_count += 1
            
            if sub_bsp_count > 0:
                features["sub_bsp_count"] = sub_bsp_count
                features["sub_buy_ratio"] = sub_buy_count / sub_bsp_count if sub_bsp_count > 0 else 0.0
                
                # 次级别信号一致性
                if current_bi.dir == BI_DIR.DOWN and sub_buy_count > sub_sell_count:
                    features["sub_level_confirmation"] = 1.0  # 下降笔中出现买点
                elif current_bi.dir == BI_DIR.UP and sub_sell_count > sub_buy_count:
                    features["sub_level_confirmation"] = 1.0  # 上升笔中出现卖点
                else:
                    features["sub_level_confirmation"] = 0.0
    
    return features 