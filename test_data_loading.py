#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据加载脚本
"""

import numpy as np
from preprocess import get_data
import matplotlib.pyplot as plt

def test_data_loading():
    """测试数据加载功能"""
    
    # 设置数据集参数
    dataset = 'BCI2a'
    data_path = 'data/BCI2a_mat/'
    subject = 0  # 第一个受试者 (0-based indexing)
    
    print("正在测试数据加载...")
    print(f"数据集: {dataset}")
    print(f"数据路径: {data_path}")
    print(f"受试者: {subject + 1}")
    
    try:
        # 加载数据
        X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot = get_data(
            data_path, subject, dataset, LOSO=False, isStandard=True
        )
        
        print("\n=== 数据加载成功 ===")
        print(f"训练数据形状: {X_train.shape}")
        print(f"训练标签形状: {y_train.shape}")
        print(f"训练标签one-hot形状: {y_train_onehot.shape}")
        print(f"测试数据形状: {X_test.shape}")
        print(f"测试标签形状: {y_test.shape}")
        print(f"测试标签one-hot形状: {y_test_onehot.shape}")
        
        # 检查数据统计
        print(f"\n=== 数据统计 ===")
        print(f"训练数据范围: [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"测试数据范围: [{X_test.min():.4f}, {X_test.max():.4f}]")
        
        # 检查标签分布
        print(f"\n=== 标签分布 ===")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        print("训练集标签分布:")
        for label, count in zip(unique_train, counts_train):
            print(f"  类别 {label}: {count} 样本")
            
        print("测试集标签分布:")
        for label, count in zip(unique_test, counts_test):
            print(f"  类别 {label}: {count} 样本")
            
        # 检查数据质量
        print(f"\n=== 数据质量检查 ===")
        train_nan = np.isnan(X_train).sum()
        test_nan = np.isnan(X_test).sum()
        train_inf = np.isinf(X_train).sum()
        test_inf = np.isinf(X_test).sum()
        
        print(f"训练数据中的NaN值: {train_nan}")
        print(f"测试数据中的NaN值: {test_nan}")
        print(f"训练数据中的无穷值: {train_inf}")
        print(f"测试数据中的无穷值: {test_inf}")
        
        if train_nan == 0 and test_nan == 0 and train_inf == 0 and test_inf == 0:
            print("✓ 数据质量检查通过")
        else:
            print("✗ 数据质量检查失败")
            
        # 检查数据维度是否符合模型要求
        print(f"\n=== 模型兼容性检查 ===")
        expected_shape = (None, 1, 22, 1001)  # (batch, 1, channels, samples)
        actual_train_shape = X_train.shape
        actual_test_shape = X_test.shape
        
        print(f"期望形状: {expected_shape}")
        print(f"训练数据实际形状: {actual_train_shape}")
        print(f"测试数据实际形状: {actual_test_shape}")
        
        if (actual_train_shape[1:] == (1, 22, 1001) and 
            actual_test_shape[1:] == (1, 22, 1001)):
            print("✓ 数据维度符合模型要求")
        else:
            print("✗ 数据维度不符合模型要求")
            print(f"  期望: (batch_size, 1, 22, 1001)")
            print(f"  实际训练: {actual_train_shape}")
            print(f"  实际测试: {actual_test_shape}")
        
        return True
        
    except Exception as e:
        print(f"\n=== 数据加载失败 ===")
        print(f"错误信息: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        
        import traceback
        print("\n完整错误追踪:")
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✓ 数据加载测试完成")
    else:
        print("\n✗ 数据加载测试失败")
