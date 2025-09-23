#!/usr/bin/env python3
"""
测试三分支架构 (CNN+Transformer+功能GCN)
"""

import numpy as np
import tensorflow as tf
from hgd_direct_loader import load_HGD_data_direct
from models import ATCNet_Transformer_GCN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_triple_branch_training():
    """测试三分支架构的完整训练流程"""
    
    print("=== 三分支架构训练测试 ===")
    
    # 设置随机种子
    tf.random.set_seed(3407)
    np.random.seed(3407)
    
    # 1. 加载数据
    print("1. 加载HGD数据...")
    data_path = 'C:/Users/徐善若/mne_data/high-gamma-dataset/data/'
    
    X_train, y_train = load_HGD_data_direct(data_path, 1, True)
    X_test, y_test = load_HGD_data_direct(data_path, 1, False)
    
    print(f"✓ 数据加载成功")
    print(f"  训练: {X_train.shape}, 标签分布: {np.bincount(y_train)}")
    print(f"  测试: {X_test.shape}, 标签分布: {np.bincount(y_test)}")
    
    # 2. 数据预处理
    print("2. 数据预处理...")
    
    # 重塑数据
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    
    # 标签转换
    y_train_onehot = to_categorical(y_train, 4)
    y_test_onehot = to_categorical(y_test, 4)
    
    # 标准化
    for ch in range(X_train.shape[2]):
        mean_val = X_train[:, 0, ch, :].mean()
        std_val = X_train[:, 0, ch, :].std()
        if std_val > 1e-8:
            X_train[:, 0, ch, :] = (X_train[:, 0, ch, :] - mean_val) / std_val
            X_test[:, 0, ch, :] = (X_test[:, 0, ch, :] - mean_val) / std_val
    
    # 分割验证集
    X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
        X_train, y_train_onehot, test_size=0.2, random_state=42
    )
    
    print(f"✓ 预处理完成: 训练{X_train.shape}, 验证{X_val.shape}")
    
    # 3. 创建三分支模型
    print("3. 创建三分支模型...")
    
    model = ATCNet_Transformer_GCN(
        n_classes=4,
        in_chans=128,
        in_samples=1000,
        # CNN参数
        eegn_F1=16, eegn_D=2,
        # Transformer参数
        transformer_d_model=64, transformer_heads=4, transformer_layers=2,
        # 功能GCN参数
        use_gcn=True, gcn_units=[32, 16], gcn_weight=0.2
    )
    
    print(f"✓ 模型创建成功: {model.count_params():,}参数")
    
    # 4. 编译和训练
    print("4. 开始训练...")
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # 添加自定义回调来显示详细进度
    from tensorflow.keras.callbacks import Callback
    
    class DetailedProgressCallback(Callback):
        def __init__(self, X_test, y_test_onehot):
            super().__init__()
            self.X_test = X_test
            self.y_test_onehot = y_test_onehot
            self.best_val_acc = 0
            
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            
            # 获取训练和验证指标
            train_acc = logs.get('accuracy', 0)
            train_loss = logs.get('loss', 0)
            val_acc = logs.get('val_accuracy', 0)
            val_loss = logs.get('val_loss', 0)
            
            # 计算测试准确率
            test_pred = self.model.predict(self.X_test, verbose=0)
            test_acc = accuracy_score(
                self.y_test_onehot.argmax(axis=1), 
                test_pred.argmax(axis=1)
            )
            
            # 记录最佳验证准确率
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                improvement = '✓'
            else:
                improvement = ' '
            
            # 详细显示
            print(f'Epoch {epoch+1:3d} {improvement} | '
                  f'Train: {train_acc:.4f}/{train_loss:.3f} | '
                  f'Val: {val_acc:.4f}/{val_loss:.3f} | '
                  f'Test: {test_acc:.4f} | '
                  f'Best: {self.best_val_acc:.4f}')
    
    # 创建回调
    progress_callback = DetailedProgressCallback(X_test, y_test_onehot)
    
    from tensorflow.keras.callbacks import EarlyStopping
    callbacks = [
        progress_callback,
        EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)  # 增加耐心值
    ]
    
    # 训练模型
    print("开始详细训练...")
    print("Epoch     | Train Acc/Loss    | Val Acc/Loss      | Test Acc | Best Val")
    print("-" * 75)
    
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=100,  # 增加到100轮
        batch_size=16,
        callbacks=callbacks,
        verbose=0  # 关闭默认输出，使用自定义显示
    )
    
    # 5. 评估
    print("5. 模型评估...")
    
    val_pred = model.predict(X_val, verbose=0)
    val_acc = accuracy_score(y_val_onehot.argmax(axis=1), val_pred.argmax(axis=1))
    
    test_pred = model.predict(X_test, verbose=0)
    test_acc = accuracy_score(y_test_onehot.argmax(axis=1), test_pred.argmax(axis=1))
    
    print(f"✓ 训练完成")
    print(f"  最佳验证准确率: {max(history.history['val_accuracy']):.4f}")
    print(f"  最终验证准确率: {val_acc:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  训练轮数: {len(history.history['val_accuracy'])}")
    
    # 6. 架构分析
    print("\\n6. 三分支架构分析:")
    print("  ✓ CNN分支: 局部时空特征提取")
    print("  ✓ Transformer分支: 全局时序依赖建模") 
    print("  ✓ 功能GCN分支: 脑区功能连接建模")
    print(f"  ✓ 特征融合: 80% CNN-Transformer + 20% GCN")
    
    return True

if __name__ == "__main__":
    try:
        success = test_triple_branch_training()
        if success:
            print("\\n🎉 三分支架构测试成功！")
    except Exception as e:
        print(f"\\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
