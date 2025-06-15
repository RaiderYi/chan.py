# 缠论与AL Brooks价格行为学特征系统概览

本文档提供了特征系统中所有实现的特征类别和具体特征的详细说明。

## 一、K线基础特征 (kl_)

### 单根K线特征
- `body_ratio` - K线实体占整个K线高度的比例
- `is_bull` - K线方向 (1:阳线, -1:阴线)
- `upper_shadow_ratio` - 上影线占K线总高度的比例
- `lower_shadow_ratio` - 下影线占K线总高度的比例
- `bar_strength` - K线强度 (收盘价在K线高度的相对位置)
- `bar_closing_position` - 收盘价位置 (1:上, 0:中, -1:下)

### 相对形态特征
- `gap_up` - 向上跳空 (1:是, 0:否)
- `gap_down` - 向下跳空 (1:是, 0:否)
- `is_outside` - 外包K线 (1:是, 0:否)
- `is_inside` - 内包K线 (1:是, 0:否)
- `price_change` - 价格变化率
- `range_change` - 振幅变化率
- `is_hh` - 高点创新高 (1:是, 0:否)
- `is_ll` - 低点创新低 (1:是, 0:否)
- `is_hl` - 更高的低点 (1:是, 0:否)
- `is_lh` - 更低的高点 (1:是, 0:否)

## 二、AL Brooks市场分类特征 (mk_)

### 突破特征 (Breakout)
- `breakout_strength` - 突破强度 (价格超出前期区间的幅度)
- `breakout_direction` - 突破方向 (1:向上, -1:向下, 0:无突破)
- `breakout_retest` - 回测突破位置 (1:上破回测, -1:下破回测, 0:无回测)
- `failed_breakout` - 失败突破 (1:上破失败, -1:下破失败, 0:无失败突破)

### 区间特征 (Trading Range)
- `range_width` - 区间宽度 (高低点价差比例)
- `range_position` - 当前价格在区间中的相对位置 (0-1)
- `upper_bound_tests` - 上边界测试次数 (归一化)
- `lower_bound_tests` - 下边界测试次数 (归一化)
- `range_maturity` - 区间成熟度 (基于测试次数)

### 通道特征 (Channel)
- `channel_direction` - 通道方向 (1:上升, -1:下降, 0:水平)
- `channel_slope` - 通道斜率 (归一化)
- `channel_width` - 通道宽度 (相对于价格)
- `channel_quality` - 通道质量 (价格与通道线的吻合度)
- `channel_position` - 当前价格在通道中的位置 (0-1)

### 通道宽窄特征
- `channel_type` - 通道类型 (1:宽通道, -1:窄通道, 0:正常)
- `channel_width_change` - 通道宽度变化
- `channel_penetration` - 价格对通道边界的突破尝试

## 三、缠论结构特征 (ch_)

### 分型特征
- `fx_type` - 分型类型 (1:顶分型, -1:底分型, 0:无分型)
- `fx_strength` - 分型强度 (分型高低点间的价格差异)

### 笔特征
- `bi_direction` - 笔方向 (1:上升, -1:下降)
- `bi_strength` - 笔强度 (起止点价格变化比例)
- `bi_length` - 笔长度 (包含的K线数量)
- `bi_slope` - 笔斜率 (价格变化/时间变化)
- `bi_position` - K线在笔中的相对位置 (0-1)
- `bi_is_sure` - 笔确定性 (1:确定, 0:未确定)
- `bi_seg_position` - 笔在线段中的位置 (0-1)

### 线段特征
- `seg_direction` - 线段方向 (1:上升, -1:下降)
- `seg_strength` - 线段强度 (起止点价格变化比例)
- `seg_bi_count` - 线段包含笔数
- `seg_is_sure` - 线段确定性 (1:确定, 0:未确定)
- `seg_eigen_ratio` - 特征序列指标

### 中枢特征
- `zs_high` - 中枢上边界价格
- `zs_low` - 中枢下边界价格
- `zs_width` - 中枢宽度 (相对价格)
- `zs_bi_count` - 中枢包含笔数
- `klu_zs_position` - K线相对中枢位置 (1:上方, 0:内部, -1:下方)
- `klu_zs_relative_pos` - 在中枢内的相对位置 (0-1)
- `zs_peak_range` - 中枢峰值范围

### 买卖点特征
- `is_bsp` - 是否买卖点 (1:是, 0:否)
- `is_buy` - 买卖方向 (1:买点, 0:卖点)
- `bsp_type` - 买卖点类型 (1-3.5)
- `bsp_divergence` - 背驰程度

## 四、趋势与波动特征 (tr_)

### 移动平均线特征
- `ma5_ratio`, `ma10_ratio`, `ma20_ratio`, `ma60_ratio` - 当前价格与MA的比率
- `ma5_slope`, `ma10_slope`, `ma20_slope`, `ma60_slope` - MA斜率
- `price_above_ma5`, `price_above_ma10`, `price_above_ma20`, `price_above_ma60` - 价格与MA关系
- `ma5_cross_ma10`, `ma5_cross_ma20` - MA交叉状态 (1:金叉, -1:死叉, 0:无交叉)
- `ma5_ma20_gap` - MA间距

### 价格趋势特征
- `trend10_direction`, `trend20_direction`, `trend60_direction` - 趋势方向
- `trend10_strength`, `trend20_strength`, `trend60_strength` - 趋势强度
- `trend10_volatility`, `trend20_volatility`, `trend60_volatility` - 趋势波动性
- `trend10_r_squared`, `trend20_r_squared`, `trend60_r_squared` - R方值 (趋势拟合优度)
- `up_streak`, `down_streak` - 趋势持续性 (同向连续K线)

### 缠论趋势特征
- `bi_direction_pattern` - 笔方向模式
- `bi_direction_consistency` - 笔方向一致性
- `seg_direction_pattern` - 线段方向模式
- `chan_trend_type` - 缠论趋势类型 (1:上升, -1:下降, 0:盘整)
- `zs_direction` - 中枢方向变化
- `zs_high_change`, `zs_low_change` - 中枢高低点变化

### 波动特征
- `volatility_5`, `volatility_10`, `volatility_20` - 短期波动率
- `volatility_change` - 波动率变化
- `price_cycle` - 波动周期
- `range_amplitude_5`, `range_amplitude_10`, `range_amplitude_20` - 波动幅度

## 五、动量与能量特征 (mo_)

### 价格动量特征
- `momentum_1`, `momentum_3`, `momentum_5`, `momentum_10`, `momentum_20` - 短期动量
- `momentum_change` - 动量变化
- `momentum_acceleration` - 动量加速度

### 交易量特征
- `volume_change` - 成交量变化
- `volume_ratio_5`, `volume_ratio_10`, `volume_ratio_20` - 量比
- `price_volume_ratio` - 价量关系
- `price_volume_sync` - 价量同步性 (1:同步, -1:背离)
- `volume_breakout` - 量能突破
- `volume_streak` - 量能积累

### 技术指标特征
- `macd_dif`, `macd_dea`, `macd_macd` - MACD值
- `macd_bar_direction` - MACD柱状图方向
- `macd_bar_change` - MACD柱状图变化
- `macd_cross` - MACD金叉/死叉
- `rsi` - RSI值
- `rsi_overbought`, `rsi_oversold` - RSI超买超卖
- `bollinger_position` - 价格在布林带中的位置
- `bollinger_width` - 布林带宽度
- `bollinger_breakout` - 布林带突破

## 六、AL Brooks交易逻辑特征 (lg_)

### 趋势后回调特征
- `pullback_depth` - 回调深度
- `pullback_quality` - 回调质量

### 趋势恢复特征
- `trend_resumption_strength` - 恢复趋势的动力
- `trend_resumption_confirmed` - 恢复趋势的确认度

### 双重顶/底形态特征
- `double_top` - 双顶形态 (1:是, 0:否)
- `double_bottom` - 双底形态 (1:是, 0:否)
- `neckline_break` - 颈线突破 (1:向上, -1:向下, 0:无)

### 趋势线突破特征
- `trendline_break` - 趋势线突破 (1:向上, -1:向下, 0:无)
- `trendline_break_strength` - 突破强度

### 交易信号强度特征
- `signal_bar_strength` - 信号K线强度
- `signal_bar_position` - 信号K线位置
- `extreme_high_position`, `extreme_low_position` - 极端位置标记
- `setup_consolidation` - 设置K线质量 (盘整度)
- `setup_direction_consistency` - 方向一致性
- `confirmation_strength` - 后续确认K线强度
- `failed_signal` - 失败信号特征

### 信号柱质量评估特征
- `signal_bar_body_score` - 信号柱实体得分
- `signal_bar_direction_score` - 信号柱方向清晰度得分
- `signal_bar_shadow_score` - 信号柱影线比例得分
- `signal_bar_position_score` - 信号柱位置得分
- `signal_bar_quality` - 信号柱总体质量得分
- `signal_bar_relative_size` - 信号柱相对大小比例

### 跟随柱强度评估特征
- `follow_through_direction_consistency` - 跟随柱与信号柱方向一致性
- `follow_through_breakthrough` - 跟随柱是否突破信号柱高低点
- `follow_through_breakthrough_size` - 跟随柱突破幅度
- `follow_through_body_ratio` - 跟随柱实体比例
- `follow_through_relative_size` - 跟随柱相对于信号柱的大小比例
- `follow_through_volume_ratio` - 跟随柱相对于信号柱的成交量比例
- `follow_through_volume_confirmation` - 成交量确认度
- `follow_through_strength` - 跟随柱总体强度得分

## 七、多时间框架特征 (tf_)

### 上级别确认特征
- `upper_bsp_confirmation` - 上级别买卖点确认
- `upper_bsp_is_buy` - 上级别买卖方向
- `upper_bsp_type` - 上级别买卖点类型
- `upper_bi_direction` - 上级别笔方向
- `upper_bi_is_sure` - 上级别笔确定性
- `upper_seg_direction` - 上级别线段方向
- `upper_seg_is_sure` - 上级别线段确定性

### 下级别细分特征
- `sub_kl_count` - 次级别K线数量
- `sub_kl_direction` - 次级别K线方向
- `sub_kl_consensus` - 次级别K线一致性
- `sub_kl_high_position`, `sub_kl_low_position` - 次级别高低点位置
- `sub_kl_path_direction` - 次级别价格路径方向
- `sub_kl_path_slope` - 次级别价格路径斜率

### 级别共振特征
- `level_resonance` - 级别共振 (1:完全共振, -1:方向相反, 0:无共振)

### 背驰与转折点特征
- `bi_divergence` - 笔背驰标记
- `bi_divergence_strength` - 笔背驰强度
- `sub_bsp_count` - 次级别买卖点数量
- `sub_buy_ratio` - 次级别买点比例
- `sub_level_confirmation` - 次级别信号确认度

## 八、价格水平特征 (pl_)

### 支撑阻力位特征
- `nearest_support_distance` - 最近支撑位距离
- `nearest_support_strength` - 最近支撑位强度
- `key_support_price` - 关键支撑价格
- `nearest_resistance_distance` - 最近阻力位距离
- `nearest_resistance_strength` - 最近阻力位强度
- `key_resistance_price` - 关键阻力价格
- `key_level_count` - 关键价格水平总数
- `price_level_position` - 价格在支撑阻力位之间的相对位置

### 价格水平测试特征
- `support_test_count` - 支撑位测试次数
- `resistance_test_count` - 阻力位测试次数
- `support_{价格}_tests` - 特定支撑位测试次数
- `resistance_{价格}_tests` - 特定阻力位测试次数

### 突破特征
- `support_breakout` - 支撑位突破标记
- `resistance_breakout` - 阻力位突破标记
- `breakout_strength` - 突破强度
- `breakout_level_price` - 突破水平价格
- `strong_breakout` - 强势突破标记
- `breakout_retest` - 回测突破位置标记

### 与缠论结构整合的价格水平特征
- `zs_support_overlap` - 中枢与支撑位重叠标记
- `zs_resistance_overlap` - 中枢与阻力位重叠标记
- `bi_reversal_level_active` - 笔转向点形成的活跃价格水平标记
- `xd_reversal_level_active` - 线段转向点形成的活跃价格水平标记
- `xd_reversal_level_distance` - 当前价格与线段转向点价格水平的距离 