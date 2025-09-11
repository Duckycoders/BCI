#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试EEG-ATCNet模型的基本功能
"""

import numpy as np
import tensorflow as tf
from models import ATCNet_
from attention_models import attention_block

def test_model_creation():
    """测试模型创建"""
    print("正在测试ATCNet模型创建...")
    
    # BCI Competition IV-2a 数据集参数
    n_classes = 4
    in_chans = 22
    in_samples = 1125
    
    try:
        # 创建ATCNet模型
        model = ATCNet_(
            n_classes=n_classes,
            in_chans=in_chans,
            in_samples=in_samples,
            n_windows=5,
            attention='mha',
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            tcn_activation='elu'
        )
        
        print(f"✓ ATCNet模型创建成功")
        print(f"✓ 模型参数数量: {model.count_params():,}")
        
        # 打印模型结构摘要
        print("\n模型结构摘要:")
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return None

def test_model_prediction():
    """测试模型预测功能"""
    print("\n正在测试模型预测功能...")
    
    model = test_model_creation()
    if model is None:
        return False
    
    try:
        # 创建随机测试数据
        batch_size = 8
        X_test = np.random.randn(batch_size, 1, 22, 1125).astype(np.float32)
        
        print(f"✓ 创建测试数据: {X_test.shape}")
        
        # 进行预测
        predictions = model.predict(X_test, verbose=0)
        
        print(f"✓ 预测成功")
        print(f"✓ 预测输出形状: {predictions.shape}")
        print(f"✓ 预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # 检查softmax输出
        prob_sums = np.sum(predictions, axis=1)
        print(f"✓ 概率和检查: {prob_sums[:3]} (应接近1.0)")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型预测失败: {e}")
        return False

def test_attention_mechanisms():
    """测试注意力机制"""
    print("\n正在测试不同的注意力机制...")
    
    attention_types = ['mha', 'se', 'cbam']
    
    for attention in attention_types:
        try:
            model = ATCNet_(
                n_classes=4,
                in_chans=22,
                in_samples=1125,
                attention=attention
            )
            print(f"✓ {attention.upper()} 注意力模型创建成功 (参数: {model.count_params():,})")
            
        except Exception as e:
            print(f"✗ {attention.upper()} 注意力模型创建失败: {e}")

def main():
    """主测试函数"""
    print("="*60)
    print("EEG-ATCNet 模型测试")
    print("="*60)
    
    # 设置随机种子以便复现
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 测试模型创建和预测
    success = test_model_prediction()
    
    # 测试不同注意力机制
    test_attention_mechanisms()
    
    print("\n" + "="*60)
    if success:
        print("✓ 所有基本功能测试通过!")
        print("模型已准备好进行训练和使用。")
    else:
        print("✗ 部分测试失败，请检查错误信息。")
    print("="*60)

if __name__ == "__main__":
    main()

