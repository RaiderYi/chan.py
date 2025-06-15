"""
特征模块测试脚本

测试新增的特征功能：
1. 信号柱质量评估
2. 跟随柱强度评估
3. 价格水平特征
"""

import sys
import os
import unittest
import numpy as np
from typing import Dict, List, Any

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KLine.KLine_Unit import CKLine_Unit
from Chan import CChan
from ChanModel.FeatureExtractor import CFeatureExtractor
from ChanModel.LogicFeatures import calculate_signal_bar_quality, calculate_follow_through_strength
from ChanModel.PriceLevelFeatures import identify_key_levels, extract_price_level_features


class TestNewFeatures(unittest.TestCase):
    """测试新增特征功能"""
    
    def setUp(self):
        """测试前准备工作"""
        # 创建模拟K线数据
        self.klus = self._create_mock_klines(100)
        
        # 模拟缠论快照
        self.chan_snapshot = self._create_mock_chan_snapshot()
        
        # 创建上下文
        self.context = {
            'bi_list': [],
            'seg_list': [],
            'zs_list': []
        }
        
        # 创建特征提取器
        self.extractor = CFeatureExtractor()
    
    def _create_mock_klines(self, count: int) -> List[CKLine_Unit]:
        """创建模拟K线数据"""
        klus = []
        
        # 创建模拟价格数据
        base_price = 100.0
        for i in range(count):
            # 模拟一个上升趋势
            price_offset = i * 0.5 + np.random.normal(0, 3)
            open_price = base_price + price_offset
            close_price = open_price + np.random.normal(0, 2)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 1))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 1))
            volume = 10000 + np.random.randint(-3000, 5000)
            
            # 创建K线对象
            klu = CKLine_Unit(idx=i, 
                             date=f"2023-01-{(i%30)+1:02d}",
                             open=open_price,
                             high=high_price,
                             low=low_price,
                             close=close_price,
                             volume=volume)
            klus.append(klu)
        
        return klus
    
    def _create_mock_chan_snapshot(self) -> CChan:
        """创建模拟缠论快照"""
        # 创建空的缠论对象
        # 这里简化处理，将其替换为简单的字典结构
        return {}
    
    def test_signal_bar_quality(self):
        """测试信号柱质量评估"""
        print("\n测试信号柱质量评估...")
        
        # 获取几个不同特征的K线进行测试
        test_cases = [
            # 大实体阳线（高质量信号柱）
            CKLine_Unit(idx=0, date="2023-01-01", open=100, high=110, low=99, close=109, volume=12000),
            # 小实体带长上影线的阳线（低质量信号柱）
            CKLine_Unit(idx=1, date="2023-01-02", open=100, high=110, low=99, close=101, volume=8000),
            # 十字星（最低质量信号柱）
            CKLine_Unit(idx=2, date="2023-01-03", open=100, high=105, low=95, close=100, volume=5000)
        ]
        
        # 使用前10根K线作为历史数据
        previous_klus = self.klus[:10]
        
        for i, klu in enumerate(test_cases):
            features = calculate_signal_bar_quality(klu, previous_klus)
            
            print(f"K线 {i+1} 信号柱特征:")
            print(f"  身体得分: {features.get('signal_bar_body_score', 0):.2f}")
            print(f"  方向得分: {features.get('signal_bar_direction_score', 0):.2f}")
            print(f"  影线得分: {features.get('signal_bar_shadow_score', 0):.2f}")
            print(f"  位置得分: {features.get('signal_bar_position_score', 0):.2f}")
            print(f"  总质量得分: {features.get('signal_bar_quality', 0):.2f}")
            
            # 确保返回了质量得分
            self.assertIn('signal_bar_quality', features)
            
            # 确保得分在0-1之间
            self.assertGreaterEqual(features['signal_bar_quality'], 0.0)
            self.assertLessEqual(features['signal_bar_quality'], 1.0)
        
        # 验证大实体K线得分高于小实体K线
        high_quality = calculate_signal_bar_quality(test_cases[0], previous_klus)
        low_quality = calculate_signal_bar_quality(test_cases[1], previous_klus)
        lowest_quality = calculate_signal_bar_quality(test_cases[2], previous_klus)
        
        self.assertGreater(high_quality['signal_bar_quality'], low_quality['signal_bar_quality'])
        self.assertGreater(low_quality['signal_bar_quality'], lowest_quality['signal_bar_quality'])
    
    def test_follow_through_strength(self):
        """测试跟随柱强度评估"""
        print("\n测试跟随柱强度评估...")
        
        # 创建信号柱和跟随柱组合进行测试
        test_cases = [
            # 1. 强信号柱 + 强跟随柱（同向）
            (CKLine_Unit(idx=0, date="2023-01-01", open=100, high=109, low=99, close=108, volume=12000),
             CKLine_Unit(idx=1, date="2023-01-02", open=107, high=115, low=106, close=114, volume=15000)),
            
            # 2. 强信号柱 + 弱跟随柱（反向）
            (CKLine_Unit(idx=2, date="2023-01-03", open=100, high=109, low=99, close=108, volume=12000),
             CKLine_Unit(idx=3, date="2023-01-04", open=108, high=109, low=102, close=103, volume=8000)),
            
            # 3. 弱信号柱 + 强跟随柱（突破）
            (CKLine_Unit(idx=4, date="2023-01-05", open=105, high=110, low=104, close=106, volume=9000),
             CKLine_Unit(idx=5, date="2023-01-06", open=107, high=115, low=106, close=113, volume=14000))
        ]
        
        # 使用前10根K线作为历史数据
        previous_klus = self.klus[:10]
        
        for i, (signal_bar, follow_bar) in enumerate(test_cases):
            features = calculate_follow_through_strength(follow_bar, signal_bar, previous_klus)
            
            print(f"测试用例 {i+1} 跟随柱特征:")
            print(f"  方向一致性: {features.get('follow_through_direction_consistency', 0):.2f}")
            print(f"  跟随柱强度: {features.get('follow_through_body_ratio', 0):.2f}")
            if 'follow_through_breakthrough' in features:
                print(f"  突破确认: {features.get('follow_through_breakthrough', 0):.2f}")
            print(f"  总体强度得分: {features.get('follow_through_strength', 0):.2f}")
            
            # 确保返回了强度得分
            self.assertIn('follow_through_strength', features)
            
            # 确保得分在0-1之间
            self.assertGreaterEqual(features['follow_through_strength'], 0.0)
            self.assertLessEqual(features['follow_through_strength'], 1.0)
        
        # 验证同向且突破的跟随柱得分最高
        strong_follow = calculate_follow_through_strength(test_cases[0][1], test_cases[0][0], previous_klus)
        weak_follow = calculate_follow_through_strength(test_cases[1][1], test_cases[1][0], previous_klus)
        
        self.assertGreater(strong_follow['follow_through_strength'], weak_follow['follow_through_strength'])
    
    def test_price_level_features(self):
        """测试价格水平特征"""
        print("\n测试价格水平特征...")
        
        # 使用真实模式的K线数据
        klu = self.klus[30]  # 选择中间的K线
        previous_klus = self.klus[:30]  # 前30根K线作为历史数据
        
        # 识别关键价格水平
        levels = identify_key_levels(klu, previous_klus)
        
        print(f"识别到 {len(levels)} 个关键价格水平:")
        for key, info in levels.items():
            print(f"  {key}: 价格={info['price']:.2f}, 触及次数={info['touches']}, 距离={info['distance']:.2f}%")
        
        # 验证能够识别支撑位和阻力位
        support_levels = [lvl for name, lvl in levels.items() if lvl['type'] == '支撑']
        resistance_levels = [lvl for name, lvl in levels.items() if lvl['type'] == '阻力']
        
        print(f"支撑位数量: {len(support_levels)}")
        print(f"阻力位数量: {len(resistance_levels)}")
        
        # 提取价格水平特征
        features = extract_price_level_features(klu, previous_klus, self.chan_snapshot, self.context)
        
        print("价格水平特征:")
        for name, value in features.items():
            print(f"  {name}: {value:.4f}")
        
        # 确保返回了相关特征
        if support_levels:
            self.assertIn("nearest_support_distance", features)
        if resistance_levels:
            self.assertIn("nearest_resistance_distance", features)


if __name__ == "__main__":
    unittest.main() 