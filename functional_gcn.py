"""
基于神经科学功能连接的GCN实现
专门用于HGD 128通道EEG数据，不依赖空间位置估计
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Lambda, Concatenate
from tensorflow.keras import backend as K

def create_functional_adjacency_matrix():
    """
    基于神经科学原理创建功能连接邻接矩阵
    不依赖空间位置，完全基于脑区功能连接模式
    """
    
    # 使用预定义的HGD 128通道电极名称，避免重复加载EDF
    electrode_names = [
        'Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
        'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5',
        'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
        'Oz', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
        'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
        'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8',
        'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'TPP9h', 'TPP10h', 'PO9', 'PO10',
        'P9', 'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
        'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h',
        'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h',
        'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h',
        'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'TPP8h', 'PPO9h', 'PPO5h',
        'PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h'
    ]
    
    n_electrodes = len(electrode_names)
    adjacency = np.eye(n_electrodes) * 0.3  # 自连接基线
    
    # 基于运动执行任务的功能脑区定义
    brain_regions = {
        'primary_motor': [],      # 主要运动皮层 (M1)
        'premotor': [],           # 前运动皮层 (PMC)
        'supplementary_motor': [], # 辅助运动区 (SMA)
        'somatosensory': [],      # 体感皮层 (S1)
        'parietal': [],           # 后顶叶皮层 (PPC)
        'frontal': [],            # 额叶皮层
        'occipital': [],          # 枕叶皮层
        'temporal': []            # 颞叶皮层
    }
    
    # 根据电极名称分配到功能脑区
    for i, electrode in enumerate(electrode_names):
        name = electrode.upper()
        
        # 主要运动皮层 (M1) - 直接控制运动
        if any(region in name for region in ['C1', 'C2', 'C3', 'C4', 'CZ']):
            brain_regions['primary_motor'].append(i)
            
        # 前运动皮层 (PMC) - 运动规划
        elif any(region in name for region in ['FC1', 'FC2', 'FC3', 'FC4', 'FCZ']):
            brain_regions['premotor'].append(i)
            
        # 辅助运动区 (SMA) - 复杂运动序列
        elif any(region in name for region in ['FC5', 'FC6', 'C5', 'C6']):
            brain_regions['supplementary_motor'].append(i)
            
        # 体感皮层 (S1) - 感觉反馈
        elif any(region in name for region in ['CP1', 'CP2', 'CP3', 'CP4', 'CPZ']):
            brain_regions['somatosensory'].append(i)
            
        # 后顶叶皮层 (PPC) - 空间处理和运动整合
        elif any(region in name for region in ['P1', 'P2', 'P3', 'P4', 'PZ']):
            brain_regions['parietal'].append(i)
            
        # 额叶皮层 - 执行控制
        elif name.startswith('F') and not name.startswith('FC'):
            brain_regions['frontal'].append(i)
            
        # 枕叶皮层 - 视觉处理
        elif name.startswith('O') or 'PO' in name:
            brain_regions['occipital'].append(i)
            
        # 颞叶皮层 - 听觉和记忆
        elif name.startswith('T') or 'TP' in name or 'FT' in name:
            brain_regions['temporal'].append(i)
    
    # 基于神经科学的功能连接强度
    connection_strengths = {
        # 运动网络内部连接 (强连接)
        ('primary_motor', 'primary_motor'): 0.9,
        ('primary_motor', 'premotor'): 0.8,
        ('primary_motor', 'supplementary_motor'): 0.7,
        ('primary_motor', 'somatosensory'): 0.8,  # 运动-感觉回路
        
        # 运动规划网络
        ('premotor', 'premotor'): 0.7,
        ('premotor', 'supplementary_motor'): 0.6,
        ('premotor', 'frontal'): 0.5,
        
        # 感觉运动整合
        ('somatosensory', 'somatosensory'): 0.7,
        ('somatosensory', 'parietal'): 0.6,
        
        # 高级认知控制
        ('parietal', 'parietal'): 0.5,
        ('parietal', 'frontal'): 0.4,
        ('frontal', 'frontal'): 0.5,
        
        # 跨模态连接（较弱）
        ('occipital', 'parietal'): 0.3,
        ('temporal', 'frontal'): 0.3,
        ('temporal', 'parietal'): 0.3
    }
    
    # 应用功能连接
    for (region1, region2), strength in connection_strengths.items():
        indices1 = brain_regions[region1]
        indices2 = brain_regions[region2]
        
        for i in indices1:
            for j in indices2:
                adjacency[i][j] = max(adjacency[i][j], strength)
                if region1 != region2:  # 确保对称性
                    adjacency[j][i] = max(adjacency[j][i], strength)
    
    # 打印脑区统计
    print("脑区电极分布:")
    for region, indices in brain_regions.items():
        if indices:
            print(f"  {region}: {len(indices)}个电极")
    
    # 行归一化
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adjacency_normalized = adjacency / row_sums
    
    return adjacency_normalized.astype(np.float32), electrode_names

class FunctionalGraphConvolution(Layer):
    """基于功能连接的图卷积层"""
    
    def __init__(self, units, adjacency_matrix, activation='relu', **kwargs):
        super(FunctionalGraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.adjacency_matrix = tf.constant(adjacency_matrix, dtype=tf.float32)
        self.activation = activation
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        super(FunctionalGraphConvolution, self).build(input_shape)
    
    def call(self, inputs):
        # 特征变换
        support = tf.matmul(inputs, self.kernel)
        
        # 功能连接图卷积
        output = tf.matmul(self.adjacency_matrix, support)
        
        # 偏置和激活
        output = tf.nn.bias_add(output, self.bias)
        
        if self.activation == 'relu':
            output = tf.nn.relu(output)
        elif self.activation == 'elu':
            output = tf.nn.elu(output)
        
        return output

def create_functional_gcn_branch(input_layer, gcn_units=[32, 16], dropout_rate=0.3):
    """
    创建基于功能连接的GCN分支
    
    Parameters:
    -----------
    input_layer : tensor
        输入 (batch, 1, 128, 1000)
    gcn_units : list
        GCN层的单元数
    dropout_rate : float
        Dropout率
    """
    
    # 创建功能连接邻接矩阵
    adjacency_matrix, electrode_names = create_functional_adjacency_matrix()
    
    # 输入变换: (batch, 1, 128, 1000) → (batch, 128, 1000)
    x = Lambda(lambda x: tf.squeeze(x, axis=1), name='gcn_squeeze')(input_layer)
    
    # 时间特征提取: (batch, 128, 1000) → (batch, 128, features)
    # 使用多种神经生理学相关的特征
    
    # 1. 功率特征（与运动相关的频段）
    x_power = Lambda(lambda x: tf.reduce_mean(tf.square(x), axis=-1, keepdims=True), 
                    name='gcn_power')(x)
    
    # 2. 变异性特征（运动准备的指标）
    x_var = Lambda(lambda x: tf.math.reduce_variance(x, axis=-1, keepdims=True), 
                  name='gcn_variance')(x)
    
    # 3. 峰值特征（事件相关电位）
    x_peak = Lambda(lambda x: tf.reduce_max(tf.abs(x), axis=-1, keepdims=True), 
                   name='gcn_peak')(x)
    
    # 组合特征: (batch, 128, 3)
    x_combined = Concatenate(axis=-1, name='gcn_feature_concat')([x_power, x_var, x_peak])
    
    # 应用功能连接GCN层
    x_gcn = x_combined
    for i, units in enumerate(gcn_units):
        x_gcn = FunctionalGraphConvolution(
            units=units,
            adjacency_matrix=adjacency_matrix,
            activation='relu',
            name=f'functional_gcn_{i+1}'
        )(x_gcn)
        
        x_gcn = BatchNormalization(name=f'gcn_bn_{i+1}')(x_gcn)
        x_gcn = Dropout(dropout_rate, name=f'gcn_dropout_{i+1}')(x_gcn)
    
    # 脑区级别的池化
    # 使用运动相关脑区的加权平均
    motor_attention = Dense(1, activation='sigmoid', name='motor_attention')(x_gcn)
    x_pooled = Lambda(
        lambda inputs: tf.reduce_sum(inputs[0] * inputs[1], axis=1),
        name='motor_weighted_pooling'
    )([x_gcn, motor_attention])
    
    return x_pooled

def test_functional_gcn():
    """测试功能连接GCN"""
    
    print("=== 测试功能连接GCN ===")
    
    # 创建邻接矩阵
    adj_matrix, electrode_names = create_functional_adjacency_matrix()
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print(f"功能连接密度: {(adj_matrix > 0.4).sum() / (128*128):.3f}")
    
    # 测试GCN分支
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    
    input_test = Input(shape=(1, 128, 1000))
    gcn_output = create_functional_gcn_branch(input_test)
    
    model_test = Model(inputs=input_test, outputs=gcn_output)
    print(f"功能GCN输出形状: {model_test.output_shape}")
    print(f"功能GCN参数量: {model_test.count_params()}")
    
    # 测试前向传播
    import numpy as np
    dummy_input = np.random.randn(4, 1, 128, 1000).astype(np.float32)
    output = model_test.predict(dummy_input, verbose=0)
    print(f"前向传播成功: {output.shape}")
    
    return adj_matrix, electrode_names

if __name__ == "__main__":
    test_functional_gcn()
