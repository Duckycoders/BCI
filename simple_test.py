#!/usr/bin/env python3
"""
简单的数据加载测试
"""

import sys
import os
sys.path.append(os.getcwd())

print("开始简单测试...")

try:
    print("1. 导入必要的库...")
    import numpy as np
    from preprocess import load_BCI2a_MOABB_data
    print("✓ 库导入成功")
    
    print("2. 测试MOABB数据加载...")
    # 测试加载一个受试者的数据
    X_train, y_train = load_BCI2a_MOABB_data('', 1, True)  # 训练数据
    X_test, y_test = load_BCI2a_MOABB_data('', 1, False)   # 测试数据
    
    print(f"✓ 数据加载成功")
    print(f"训练数据形状: {X_train.shape}")
    print(f"训练标签形状: {y_train.shape}")
    print(f"测试数据形状: {X_test.shape}")  
    print(f"测试标签形状: {y_test.shape}")
    
    print("3. 检查标签分布...")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    print("训练集:")
    for label, count in zip(unique_train, counts_train):
        print(f"  类别 {label}: {count} 样本")
        
    print("测试集:")
    for label, count in zip(unique_test, counts_test):
        print(f"  类别 {label}: {count} 样本")
    
    # 检查数据是否太少
    total_train = len(y_train)
    total_test = len(y_test)
    print(f"\n总训练样本: {total_train}")
    print(f"总测试样本: {total_test}")
    
    if total_train < 100:
        print("⚠️  警告: 训练样本数量可能太少!")
    if total_test < 50:
        print("⚠️  警告: 测试样本数量可能太少!")
        
    print("✓ 简单测试完成")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

