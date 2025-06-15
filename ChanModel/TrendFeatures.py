"""
趋势、波动和动量特征提取模块

本模块实现了:
1. 趋势特征提取
2. 波动特征提取
3. 动量和能量特征提取
"""

import numpy as np
from typing import Dict, List, Any, Optional

from KLine.KLine_Unit import CKLine_Unit
from Common.CEnum import BI_DIR


def extract_trend_features(klu: CKLine_Unit, previous_klus: List[CKLine_Unit], 
                           context: Dict[str, Any]) -> Dict[str, float]:
    """
    提取趋势和波动特征
    
    Args:
        klu: 当前K线
        previous_klus: 前N根K线列表
        context: 上下文信息（笔、线段、中枢等）
        
    Returns:
        趋势和波动特征字典
    """
    features = {}
    
    # 安全检查
    if not previous_klus:
        return features
    
    # 收集价格序列
    closes = [klu.close] + [k.close for k in previous_klus]
    highs = [klu.high] + [k.high for k in previous_klus]
    lows = [klu.low] + [k.low for k in previous_klus]
    
    # --- 移动平均线特征 ---
    # 计算不同周期的移动平均
    for period in [5, 10, 20, 60]:
        if len(closes) >= period:
            ma = sum(closes[:period]) / period
            features[f"ma{period}_ratio"] = klu.close / ma - 1 if ma > 0 else 0.0
            
            # MA斜率
            if len(closes) >= period + 5:
                ma_prev = sum(closes[5:period+5]) / period
                features[f"ma{period}_slope"] = (ma - ma_prev) / ma_prev if ma_prev > 0 else 0.0
            
            # 价格与MA关系
            if klu.close > ma:
                features[f"price_above_ma{period}"] = 1.0
            else:
                features[f"price_above_ma{period}"] = -1.0
    
    # MA交叉
    if len(closes) >= 20:
        ma5 = sum(closes[:5]) / 5
        ma10 = sum(closes[:10]) / 10
        ma20 = sum(closes[:20]) / 20
        
        ma5_prev = sum(closes[1:6]) / 5
        ma10_prev = sum(closes[1:11]) / 10
        ma20_prev = sum(closes[1:21]) / 20
        
        # 5日均线与10日均线交叉
        if ma5 > ma10 and ma5_prev <= ma10_prev:
            features["ma5_cross_ma10"] = 1.0  # 金叉
        elif ma5 < ma10 and ma5_prev >= ma10_prev:
            features["ma5_cross_ma10"] = -1.0  # 死叉
        else:
            features["ma5_cross_ma10"] = 0.0  # 无交叉
        
        # 5日均线与20日均线交叉
        if ma5 > ma20 and ma5_prev <= ma20_prev:
            features["ma5_cross_ma20"] = 1.0  # 金叉
        elif ma5 < ma20 and ma5_prev >= ma20_prev:
            features["ma5_cross_ma20"] = -1.0  # 死叉
        else:
            features["ma5_cross_ma20"] = 0.0  # 无交叉
        
        # MA间距
        features["ma5_ma20_gap"] = ma5 / ma20 - 1 if ma20 > 0 else 0.0
    
    # --- 价格趋势特征 ---
    # 使用线性回归计算趋势方向和强度
    for period in [10, 20, 60]:
        if len(closes) >= period:
            x = np.arange(period)
            y = np.array(closes[:period])
            
            # 线性回归
            slope, intercept = np.polyfit(x, y, 1)
            
            # 趋势方向
            features[f"trend{period}_direction"] = np.sign(slope)
            
            # 趋势强度（归一化斜率）
            features[f"trend{period}_strength"] = slope / np.mean(y) if np.mean(y) > 0 else 0.0
            
            # 趋势波动性（价格围绕趋势线的标准差）
            trend_line = intercept + slope * x
            deviation = y - trend_line
            features[f"trend{period}_volatility"] = np.std(deviation) / np.mean(y) if np.mean(y) > 0 else 0.0
            
            # R方值（趋势拟合优度）
            correlation_matrix = np.corrcoef(x, y)
            correlation_xy = correlation_matrix[0, 1]
            r_squared = correlation_xy**2
            features[f"trend{period}_r_squared"] = r_squared
    
    # 趋势持续性（同向连续K线数量）
    up_streak = 0
    down_streak = 0
    
    for i in range(1, min(20, len(closes))):
        if closes[i-1] > closes[i]:
            up_streak = 0
            down_streak += 1
        elif closes[i-1] < closes[i]:
            down_streak = 0
            up_streak += 1
        else:
            # 价格不变，保持当前连续计数
            pass
    
    features["up_streak"] = min(up_streak / 10.0, 1.0)  # 归一化
    features["down_streak"] = min(down_streak / 10.0, 1.0)  # 归一化
    
    # --- 缠论趋势特征 ---
    # 笔方向序列
    bi_list = context.get('bi_list', [])
    if len(bi_list) >= 3:
        # 获取最近3笔的方向
        recent_bis = bi_list[-3:]
        directions = [1.0 if bi.dir == BI_DIR.UP else -1.0 for bi in recent_bis]
        
        # 笔方向模式
        features["bi_direction_pattern"] = directions[0] * 4 + directions[1] * 2 + directions[2]
        
        # 笔方向一致性
        if all(d == directions[0] for d in directions):
            features["bi_direction_consistency"] = 1.0
        else:
            # 计算方向变化次数
            changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
            features["bi_direction_consistency"] = 1.0 - changes / (len(directions) - 1)
    
    # 线段方向序列
    seg_list = context.get('seg_list', [])
    if len(seg_list) >= 3:
        # 获取最近3段的方向
        recent_segs = seg_list[-3:]
        seg_directions = [1.0 if seg.dir == BI_DIR.UP else -1.0 for seg in recent_segs]
        
        # 线段方向模式
        features["seg_direction_pattern"] = seg_directions[0] * 4 + seg_directions[1] * 2 + seg_directions[2]
        
        # 判断趋势类型
        if all(d == 1.0 for d in seg_directions):
            features["chan_trend_type"] = 1.0  # 上升趋势
        elif all(d == -1.0 for d in seg_directions):
            features["chan_trend_type"] = -1.0  # 下降趋势
        else:
            features["chan_trend_type"] = 0.0  # 盘整趋势
    
    # 中枢方向变化
    zs_list = context.get('zs_list', [])
    if len(zs_list) >= 2:
        last_zs = zs_list[-1]
        prev_zs = zs_list[-2]
        
        # 中枢移动方向
        if last_zs.low > prev_zs.high:
            features["zs_direction"] = 1.0  # 中枢上移
        elif last_zs.high < prev_zs.low:
            features["zs_direction"] = -1.0  # 中枢下移
        else:
            features["zs_direction"] = 0.0  # 中枢重叠
        
        # 中枢高低点变化
        if prev_zs.high > 0:
            features["zs_high_change"] = (last_zs.high - prev_zs.high) / prev_zs.high
        else:
            features["zs_high_change"] = 0.0
            
        if prev_zs.low > 0:
            features["zs_low_change"] = (last_zs.low - prev_zs.low) / prev_zs.low
        else:
            features["zs_low_change"] = 0.0
    
    # --- 波动特征 ---
    # 短期波动率（价格标准差/均值）
    for period in [5, 10, 20]:
        if len(closes) >= period:
            price_std = np.std(closes[:period])
            price_mean = np.mean(closes[:period])
            features[f"volatility_{period}"] = price_std / price_mean if price_mean > 0 else 0.0
    
    # 波动率变化
    if len(closes) >= 25:
        vol_5_recent = features.get("volatility_5", 0.0)
        vol_5_prev = np.std(closes[20:25]) / np.mean(closes[20:25]) if np.mean(closes[20:25]) > 0 else 0.0
        
        features["volatility_change"] = vol_5_recent / vol_5_prev - 1 if vol_5_prev > 0 else 0.0
    
    # 波动周期（使用自相关函数估计，简化版）
    if len(closes) >= 40:
        returns = [closes[i]/closes[i+1]-1 for i in range(len(closes)-1)][:40]
        autocorr = []
        mean_return = np.mean(returns)
        
        for lag in range(1, 21):
            numerator = sum((returns[i] - mean_return) * (returns[i+lag] - mean_return) for i in range(len(returns)-lag))
            denominator = sum((r - mean_return)**2 for r in returns)
            if denominator > 0:
                autocorr.append(numerator / denominator)
            else:
                autocorr.append(0)
        
        # 找到第一个局部最大值作为周期估计
        cycle = 1
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                cycle = i + 1
                break
        
        features["price_cycle"] = cycle / 20.0  # 归一化
    
    # 波动幅度（最近N天价格的最大振幅）
    for period in [5, 10, 20]:
        if len(closes) >= period:
            max_price = max(highs[:period])
            min_price = min(lows[:period])
            
            if min_price > 0:
                features[f"range_amplitude_{period}"] = (max_price - min_price) / min_price
            else:
                features[f"range_amplitude_{period}"] = 0.0
    
    return features


def extract_momentum_features(klu: CKLine_Unit, previous_klus: List[CKLine_Unit]) -> Dict[str, float]:
    """
    提取动量和能量特征
    
    Args:
        klu: 当前K线
        previous_klus: 前N根K线列表
        
    Returns:
        动量和能量特征字典
    """
    features = {}
    
    # 安全检查
    if not previous_klus:
        return features
    
    # 收集价格序列
    closes = [klu.close] + [k.close for k in previous_klus]
    
    # --- 价格动量特征 ---
    # 短期动量（不同周期的价格变化率）
    for period in [1, 3, 5, 10, 20]:
        if len(closes) > period:
            features[f"momentum_{period}"] = closes[0] / closes[period] - 1
    
    # 动量变化（动量加速/减速）
    if len(closes) >= 11:
        mom_5 = closes[0] / closes[5] - 1
        mom_5_prev = closes[5] / closes[10] - 1
        
        if mom_5_prev != 0:
            features["momentum_change"] = mom_5 / mom_5_prev - 1
        else:
            features["momentum_change"] = 0.0
        
        # 动量加速度（三阶导数近似）
        if len(closes) >= 16:
            mom_5_prev2 = closes[10] / closes[15] - 1
            if mom_5_prev != mom_5_prev2:
                features["momentum_acceleration"] = (mom_5 - mom_5_prev) - (mom_5_prev - mom_5_prev2)
            else:
                features["momentum_acceleration"] = 0.0
    
    # --- 交易量特征 ---
    # 检查是否有成交量数据
    if hasattr(klu, 'volume') and len(previous_klus) > 0 and hasattr(previous_klus[0], 'volume'):
        volumes = [klu.volume] + [k.volume for k in previous_klus if hasattr(k, 'volume')]
        
        # 成交量变化
        if len(volumes) > 1 and volumes[1] > 0:
            features["volume_change"] = volumes[0] / volumes[1] - 1
        
        # 量比（当前成交量相对N周期平均的比率）
        for period in [5, 10, 20]:
            if len(volumes) > period:
                avg_volume = sum(volumes[1:period+1]) / period
                if avg_volume > 0:
                    features[f"volume_ratio_{period}"] = volumes[0] / avg_volume
                else:
                    features[f"volume_ratio_{period}"] = 1.0
        
        # 价量关系
        if len(closes) > 1 and len(volumes) > 1:
            price_change = closes[0] / closes[1] - 1
            volume_change = features.get("volume_change", 0.0)
            
            if volume_change != 0:
                features["price_volume_ratio"] = price_change / volume_change
            else:
                features["price_volume_ratio"] = 0.0
            
            # 价量同步性
            if (price_change > 0 and volume_change > 0) or (price_change < 0 and volume_change < 0):
                features["price_volume_sync"] = 1.0
            else:
                features["price_volume_sync"] = -1.0
        
        # 量能突破（成交量突破前期高点）
        if len(volumes) >= 20:
            max_vol_20 = max(volumes[1:21])
            if volumes[0] > max_vol_20:
                features["volume_breakout"] = volumes[0] / max_vol_20 - 1
            else:
                features["volume_breakout"] = 0.0
            
            # 量能积累（连续放量）
            vol_streak = 0
            for i in range(1, min(5, len(volumes))):
                if volumes[i-1] > volumes[i]:
                    vol_streak += 1
                else:
                    break
            
            features["volume_streak"] = vol_streak / 5.0  # 归一化
    
    # --- 技术指标特征 ---
    # MACD值
    if hasattr(klu, 'macd') and hasattr(klu.macd, 'dif') and hasattr(klu.macd, 'dea'):
        features["macd_dif"] = klu.macd.dif
        features["macd_dea"] = klu.macd.dea
        features["macd_macd"] = klu.macd.macd
        
        # MACD柱状图方向
        if klu.macd.macd > 0:
            features["macd_bar_direction"] = 1.0
        else:
            features["macd_bar_direction"] = -1.0
        
        # MACD柱状图变化
        if len(previous_klus) > 0 and hasattr(previous_klus[0], 'macd') and hasattr(previous_klus[0].macd, 'macd'):
            prev_macd = previous_klus[0].macd.macd
            if prev_macd != 0:
                features["macd_bar_change"] = klu.macd.macd / prev_macd - 1
            else:
                features["macd_bar_change"] = 0.0
        
        # MACD金叉/死叉
        if len(previous_klus) > 0 and hasattr(previous_klus[0], 'macd') and hasattr(previous_klus[0].macd, 'dif') and hasattr(previous_klus[0].macd, 'dea'):
            if klu.macd.dif > klu.macd.dea and previous_klus[0].macd.dif <= previous_klus[0].macd.dea:
                features["macd_cross"] = 1.0  # 金叉
            elif klu.macd.dif < klu.macd.dea and previous_klus[0].macd.dif >= previous_klus[0].macd.dea:
                features["macd_cross"] = -1.0  # 死叉
            else:
                features["macd_cross"] = 0.0
    
    # RSI值
    if hasattr(klu, 'rsi'):
        features["rsi"] = klu.rsi
        
        # RSI超买超卖
        if klu.rsi > 70:
            features["rsi_overbought"] = 1.0
        elif klu.rsi < 30:
            features["rsi_oversold"] = 1.0
        else:
            features["rsi_overbought"] = 0.0
            features["rsi_oversold"] = 0.0
    
    # 布林带位置
    if hasattr(klu, 'boll') and hasattr(klu.boll, 'upper') and hasattr(klu.boll, 'lower') and hasattr(klu.boll, 'mid'):
        # 价格在布林带中的相对位置
        band_width = klu.boll.upper - klu.boll.lower
        if band_width > 0:
            position = (klu.close - klu.boll.lower) / band_width
            features["bollinger_position"] = position
        else:
            features["bollinger_position"] = 0.5
        
        # 布林带宽度
        if klu.boll.mid > 0:
            features["bollinger_width"] = band_width / klu.boll.mid
        else:
            features["bollinger_width"] = 0.0
        
        # 布林带突破
        if klu.close > klu.boll.upper:
            features["bollinger_breakout"] = 1.0  # 上突破
        elif klu.close < klu.boll.lower:
            features["bollinger_breakout"] = -1.0  # 下突破
        else:
            features["bollinger_breakout"] = 0.0  # 无突破
    
    return features 