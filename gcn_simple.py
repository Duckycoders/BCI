""" 
Simple and stable Graph Convolutional Network for EEG data
Designed to be non-intrusive to existing ATCNet architecture
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Lambda, GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate
from tensorflow.keras import backend as K

def create_eeg_adjacency_matrix():
    """
    Create a simple adjacency matrix for 22 EEG electrodes
    Based on standard 10-20 system spatial relationships
    """
    # 22个电极的邻接关系（简化版本）
    n_electrodes = 22
    adjacency = np.eye(n_electrodes, dtype=np.float32)  # 自连接
    
    # 基于常见的EEG电极邻接关系手动定义
    # 这里使用一个简化的邻接模式，主要连接相邻的电极
    connections = [
        # 中线电极连接
        (0, 3), (3, 9), (9, 15), (15, 19), (19, 21),  # Fz-FCz-Cz-CPz-Pz-POz
        
        # 左侧电极连接  
        (1, 2), (2, 8), (8, 14), (14, 18),  # FC3-FC1-C1-CP1-P1
        (6, 7), (7, 8), (13, 14),  # C5-C3-C1, CP3-CP1
        
        # 右侧电极连接
        (4, 5), (4, 10), (10, 16), (16, 20),  # FC2-FC4-C2-CP2-P2  
        (10, 11), (11, 12), (16, 17),  # C2-C4-C6, CP2-CP4
        
        # 横向连接
        (1, 3), (3, 4), (2, 3), (3, 4),  # 额区
        (7, 9), (9, 11), (8, 9), (9, 10),  # 中央区
        (13, 15), (15, 17), (14, 15), (15, 16),  # 顶区
    ]
    
    # 填充邻接矩阵
    for i, j in connections:
        if i < n_electrodes and j < n_electrodes:
            adjacency[i][j] = 0.7  # 邻接权重
            adjacency[j][i] = 0.7  # 对称
    
    # 归一化（行归一化）
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-8)
    
    return adjacency

def simple_gcn_layer(x, adjacency_matrix, units, activation='relu', dropout_rate=0.3, name_prefix='gcn'):
    """
    简单的图卷积层实现
    
    Parameters:
    -----------
    x : tensor
        输入特征 (batch, nodes, features)
    adjacency_matrix : tensor
        邻接矩阵 (nodes, nodes)
    units : int
        输出特征维度
    """
    
    # 特征变换
    x_transformed = Dense(units, use_bias=False, name=f'{name_prefix}_transform')(x)
    
    # 图卷积：A @ X @ W
    # 使用Lambda层来应用矩阵乘法
    adj_tensor = tf.constant(adjacency_matrix, dtype=tf.float32)
    x_conv = Lambda(lambda inputs: tf.matmul(adj_tensor, inputs), name=f'{name_prefix}_conv')(x_transformed)
    
    # 归一化和激活
    x_norm = BatchNormalization(name=f'{name_prefix}_bn')(x_conv)
    x_act = Activation(activation, name=f'{name_prefix}_act')(x_norm)
    x_drop = Dropout(dropout_rate, name=f'{name_prefix}_dropout')(x_act)
    
    return x_drop

def simple_gcn_branch(input_layer, adjacency_matrix=None, name_prefix='gcn_branch'):
    """
    简单的GCN分支，专门提取空间特征
    
    Parameters:
    -----------
    input_layer : tensor
        输入 (batch, 1, channels, time_samples)
    adjacency_matrix : numpy.ndarray
        电极邻接矩阵
    
    Returns:
    --------
    tensor : GCN分支的输出特征
    """
    
    if adjacency_matrix is None:
        adjacency_matrix = create_eeg_adjacency_matrix()
    
    # 输入形状转换: (batch, 1, channels, time) → (batch, channels, time)
    x = Lambda(lambda x: tf.squeeze(x, axis=1), name=f'{name_prefix}_squeeze')(input_layer)
    
    # 时间维度聚合: (batch, channels, time) → (batch, channels)
    # 使用简单的全局平均池化
    x = Lambda(lambda x: tf.reduce_mean(x, axis=-1), name=f'{name_prefix}_temporal_pool')(x)
    
    # 第一层GCN: (batch, channels) → (batch, channels, 16)
    x = tf.expand_dims(x, axis=-1)  # (batch, channels, 1)
    x = simple_gcn_layer(x, adjacency_matrix, units=16, name_prefix=f'{name_prefix}_1')
    
    # 第二层GCN: (batch, channels, 16) → (batch, channels, 8)  
    x = simple_gcn_layer(x, adjacency_matrix, units=8, name_prefix=f'{name_prefix}_2')
    
    # 全局池化: (batch, channels, 8) → (batch, 8)
    x = Lambda(lambda x: tf.reduce_mean(x, axis=1), name=f'{name_prefix}_global_pool')(x)
    
    return x

def create_atcnet_with_gcn_branch(n_classes, in_chans=22, in_samples=1001, 
                                 use_gcn=True, gcn_weight=0.3):
    """
    创建带有GCN分支的ATCNet，保持原架构不变
    
    Parameters:
    -----------
    n_classes : int
        分类数量
    in_chans : int  
        EEG通道数
    in_samples : int
        时间采样点数
    use_gcn : bool
        是否使用GCN分支
    gcn_weight : float
        GCN分支的权重 (0-1)
    """
    
    from models import ATCNet_
    from tensorflow.keras.layers import Input, Dense, Dropout, Add
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import L2
    
    # 创建输入
    input_layer = Input(shape=(1, in_chans, in_samples))
    
    # 原始ATCNet分支（保持完全不变）
    atcnet_model = ATCNet_(n_classes=n_classes, in_chans=in_chans, in_samples=in_samples)
    
    # 获取ATCNet的倒数第二层特征（在softmax之前）
    # 我们需要修改ATCNet来提取特征
    atcnet_features = atcnet_model.layers[-2](atcnet_model.layers[-3].output)  # 获取softmax前的特征
    
    if use_gcn:
        # GCN分支
        gcn_features = simple_gcn_branch(input_layer, name_prefix='gcn')
        
        # 特征融合（简单的加权组合）
        # 将GCN特征映射到相同维度
        gcn_mapped = Dense(n_classes, name='gcn_projection')(gcn_features)
        
        # 加权融合
        fused_features = Lambda(
            lambda inputs: (1 - gcn_weight) * inputs[0] + gcn_weight * inputs[1],
            name='feature_fusion'
        )([atcnet_features, gcn_mapped])
        
    else:
        fused_features = atcnet_features
    
    # 最终输出
    output = Activation('softmax', name='final_output')(fused_features)
    
    return Model(inputs=input_layer, outputs=output)

