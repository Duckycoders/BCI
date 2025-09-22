#!/usr/bin/env python3
"""
直接测试HGD数据集训练，绕过复杂的配置
"""

import numpy as np
import tensorflow as tf
from hgd_direct_loader import load_HGD_data_direct
from models import ATCNet_Transformer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_hgd_training():
    """直接测试HGD训练流程"""
    
    print("=== HGD训练测试 ===")
    
    # 设置随机种子
    tf.random.set_seed(3407)
    np.random.seed(3407)
    
    # 数据配置
    data_path = 'C:/Users/徐善若/mne_data/high-gamma-dataset/data/'
    
    print("1. 加载HGD数据...")
    try:
        # 加载第一个受试者的数据
        X_train, y_train = load_HGD_data_direct(data_path, 1, True)
        X_test, y_test = load_HGD_data_direct(data_path, 1, False)
        
        print(f"✓ 数据加载成功")
        print(f"  训练: {X_train.shape}, 标签: {np.bincount(y_train)}")
        print(f"  测试: {X_test.shape}, 标签: {np.bincount(y_test)}")
        
        # 数据预处理
        print("2. 数据预处理...")
        
        # 重塑为模型期望的格式: (trials, 1, channels, time)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
        
        # 转换标签为one-hot
        y_train_onehot = to_categorical(y_train, 4)
        y_test_onehot = to_categorical(y_test, 4)
        
        # 简单标准化
        for ch in range(X_train.shape[2]):
            mean_val = X_train[:, 0, ch, :].mean()
            std_val = X_train[:, 0, ch, :].std()
            X_train[:, 0, ch, :] = (X_train[:, 0, ch, :] - mean_val) / (std_val + 1e-8)
            X_test[:, 0, ch, :] = (X_test[:, 0, ch, :] - mean_val) / (std_val + 1e-8)
        
        # 分割验证集
        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
            X_train, y_train_onehot, test_size=0.2, random_state=42
        )
        
        print(f"✓ 预处理完成")
        print(f"  训练: {X_train.shape}")
        print(f"  验证: {X_val.shape}")
        print(f"  测试: {X_test.shape}")
        
        # 创建模型
        print("3. 创建ATCNet_Transformer模型...")
        model = ATCNet_Transformer(
            n_classes=4,
            in_chans=128,  # HGD的128通道
            in_samples=1000,  # HGD的1000样本
            transformer_d_model=64,
            transformer_heads=4,
            transformer_layers=2
        )
        
        print(f"✓ 模型创建成功")
        print(f"  参数量: {model.count_params():,}")
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        print("4. 开始训练...")
        
        # 训练模型
        history = model.fit(
            X_train, y_train_onehot,
            validation_data=(X_val, y_val_onehot),
            epochs=50,
            batch_size=16,
            verbose=1
        )
        
        # 评估
        print("5. 模型评估...")
        val_pred = model.predict(X_val, verbose=0)
        val_acc = accuracy_score(y_val_onehot.argmax(axis=1), val_pred.argmax(axis=1))
        
        test_pred = model.predict(X_test, verbose=0)
        test_acc = accuracy_score(y_test_onehot.argmax(axis=1), test_pred.argmax(axis=1))
        
        print(f"✓ 训练完成")
        print(f"  最佳验证准确率: {max(history.history['val_accuracy']):.4f}")
        print(f"  最终验证准确率: {val_acc:.4f}")
        print(f"  测试准确率: {test_acc:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hgd_training()
    if success:
        print("\n🎉 HGD训练测试成功！")
    else:
        print("\n❌ HGD训练测试失败")
