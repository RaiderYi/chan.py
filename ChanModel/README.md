# 缠论与AL Brooks价格行为学特征系统

本特征系统融合了缠论和AL Brooks价格行为学的核心概念，构建了一个全面的特征体系，用于量化交易策略的开发和评估。

## 特征体系架构

特征系统包含七大类特征：

1. **K线基础特征** - 描述单个K线的基本属性和相对关系
2. **AL Brooks市场分类特征** - 识别市场环境（突破、区间、通道）
3. **缠论结构特征** - 提取缠论框架中的分型、笔、线段、中枢和买卖点信息
4. **趋势与波动特征** - 捕捉多周期趋势方向、强度和波动特性
5. **动量与能量特征** - 提取价格动量和交易量特征
6. **AL Brooks交易逻辑特征** - 分析入场信号质量和交易逻辑
7. **多时间框架特征** - 融合不同时间周期的信息

## 使用方法

### 1. 基本使用

```python
from ChanModel.FeatureExtractor import CFeatureExtractor

# 初始化特征提取器
feature_extractor = CFeatureExtractor()

# 提取特定K线的所有特征
features = feature_extractor.extract_all_features(chan_snapshot, klu_idx)

# 使用特征进行模型训练或决策
print(features)
```

### 2. 提取特定类别特征

如果只需要某一类特征，可以直接调用相应的特征提取函数：

```python
from ChanModel.KlineFeatures import extract_kline_basic_features
from ChanModel.ChanFeatures import extract_chan_structure_features

# 获取K线基础特征
kl_features = extract_kline_basic_features(klu, previous_klus)

# 获取缠论结构特征
context = {...}  # 包含当前笔、线段、中枢的上下文信息
chan_features = extract_chan_structure_features(klu, context)
```

### 3. 机器学习模型训练

参考 `Debug/enhanced_strategy_demo5.py` 文件，了解如何：
- 收集买卖点特征
- 生成训练样本
- 训练XGBoost模型
- 分析特征重要性
- 评估模型性能

## 特征前缀说明

为了便于区分和管理不同类别的特征，系统使用了以下前缀：

- `kl_` - K线基础特征
- `mk_` - AL Brooks市场分类特征
- `ch_` - 缠论结构特征
- `tr_` - 趋势与波动特征
- `mo_` - 动量与能量特征
- `lg_` - AL Brooks交易逻辑特征
- `tf_` - 多时间框架特征

## 特征系统优势

1. **多维度分析** - 同时考虑价格行为、趋势结构和成交量
2. **层次化设计** - 从微观K线特征到宏观市场环境分类
3. **跨周期整合** - 融合不同时间框架的信号
4. **预测导向** - 专注于买卖点的识别和确认

## 扩展与定制

特征系统设计为可扩展架构，可以根据需要添加新的特征：

1. 在相应的特征模块中添加新的特征提取函数
2. 在 `FeatureExtractor.extract_all_features()` 方法中集成新特征
3. 为新特征添加适当的前缀，以便于分类管理

## 示例应用

1. **买卖点质量评估** - 预测缠论买卖点的有效性
2. **市场环境分类** - 识别当前市场是处于趋势、区间还是变盘阶段
3. **多因子策略开发** - 基于最重要的特征构建因子模型
4. **机器学习模型训练** - 使用完整特征集训练预测模型 