"""
基于HGD数据集真实128通道EEG电极的GCN实现
使用实际的电极名称和位置构建图卷积网络
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Lambda, GlobalAveragePooling1D, Concatenate
from tensorflow.keras import backend as K

def get_real_hgd_electrode_positions():
    """
    基于HGD数据集实际电极名称的位置映射
    使用标准10-20系统和高密度扩展的真实坐标
    """
    
    # HGD数据集的实际电极名称（前128个EEG通道）
    real_electrodes = [
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
    
    # 标准10-20系统位置 + 高密度扩展位置
    # 坐标系：(x, y) 其中x为左右(-1到1)，y为前后(-1到1)
    position_map = {
        # 标准10-20电极
        'Fpz': (0.0, 0.9), 'Fp1': (-0.3, 0.85), 'Fp2': (0.3, 0.85),
        'AFz': (0.0, 0.8), 'AF3': (-0.25, 0.8), 'AF4': (0.25, 0.8),
        'AF7': (-0.55, 0.75), 'AF8': (0.55, 0.75),
        'Fz': (0.0, 0.7), 'F1': (-0.2, 0.7), 'F2': (0.2, 0.7),
        'F3': (-0.4, 0.7), 'F4': (0.4, 0.7), 'F5': (-0.6, 0.65), 'F6': (0.6, 0.65),
        'F7': (-0.75, 0.6), 'F8': (0.75, 0.6),
        'FCz': (0.0, 0.5), 'FC1': (-0.2, 0.5), 'FC2': (0.2, 0.5),
        'FC3': (-0.4, 0.5), 'FC4': (0.4, 0.5), 'FC5': (-0.6, 0.45), 'FC6': (0.6, 0.45),
        'FT7': (-0.75, 0.3), 'FT8': (0.75, 0.3), 'FT9': (-0.85, 0.2), 'FT10': (0.85, 0.2),
        'Cz': (0.0, 0.0), 'C1': (-0.2, 0.0), 'C2': (0.2, 0.0),
        'C3': (-0.4, 0.0), 'C4': (0.4, 0.0), 'C5': (-0.6, 0.0), 'C6': (0.6, 0.0),
        'T7': (-0.75, 0.0), 'T8': (0.75, 0.0), 'M1': (-0.9, 0.0), 'M2': (0.9, 0.0),
        'CPz': (0.0, -0.4), 'CP1': (-0.2, -0.4), 'CP2': (0.2, -0.4),
        'CP3': (-0.4, -0.4), 'CP4': (0.4, -0.4), 'CP5': (-0.6, -0.45), 'CP6': (0.6, -0.45),
        'TP7': (-0.75, -0.3), 'TP8': (0.75, -0.3),
        'Pz': (0.0, -0.7), 'P1': (-0.2, -0.7), 'P2': (0.2, -0.7),
        'P3': (-0.4, -0.7), 'P4': (0.4, -0.7), 'P5': (-0.6, -0.65), 'P6': (0.6, -0.65),
        'P7': (-0.75, -0.6), 'P8': (0.75, -0.6), 'P9': (-0.85, -0.5), 'P10': (0.85, -0.5),
        'POz': (0.0, -0.8), 'PO3': (-0.25, -0.8), 'PO4': (0.25, -0.8),
        'PO5': (-0.45, -0.8), 'PO6': (0.45, -0.8), 'PO7': (-0.6, -0.75), 'PO8': (0.6, -0.75),
        'PO9': (-0.75, -0.7), 'PO10': (0.75, -0.7),
        'Oz': (0.0, -0.9), 'O1': (-0.3, -0.85), 'O2': (0.3, -0.85),
        
        # 高密度电极位置估计
        'AFF1': (-0.15, 0.85), 'AFF2': (0.15, 0.85), 'AFF5h': (-0.5, 0.8), 'AFF6h': (0.5, 0.8),
        'FFC1h': (-0.15, 0.6), 'FFC2h': (0.15, 0.6), 'FFC3h': (-0.3, 0.6), 'FFC4h': (0.3, 0.6),
        'FFC5h': (-0.5, 0.55), 'FFC6h': (0.5, 0.55), 'FFT7h': (-0.7, 0.5), 'FFT8h': (0.7, 0.5),
        'FCC1h': (-0.15, 0.35), 'FCC2h': (0.15, 0.35), 'FCC3h': (-0.3, 0.35), 'FCC4h': (0.3, 0.35),
        'FCC5h': (-0.5, 0.3), 'FCC6h': (0.5, 0.3), 'FTT7h': (-0.7, 0.15), 'FTT8h': (0.7, 0.15),
        'FTT9h': (-0.8, 0.1), 'FTT10h': (0.8, 0.1), 'TTP7h': (-0.7, -0.15), 'TTP8h': (0.7, -0.15),
        'CCP1h': (-0.15, -0.25), 'CCP2h': (0.15, -0.25), 'CCP3h': (-0.3, -0.25), 'CCP4h': (0.3, -0.25),
        'CCP5h': (-0.5, -0.3), 'CCP6h': (0.5, -0.3), 'TPP7h': (-0.7, -0.4), 'TPP8h': (0.7, -0.4),
        'TPP9h': (-0.8, -0.35), 'TPP10h': (0.8, -0.35), 'CPP1h': (-0.15, -0.55), 'CPP2h': (0.15, -0.55),
        'CPP3h': (-0.3, -0.55), 'CPP4h': (0.3, -0.55), 'CPP5h': (-0.5, -0.6), 'CPP6h': (0.5, -0.6),
        'PPO1': (-0.15, -0.75), 'PPO2': (0.15, -0.75), 'PPO5h': (-0.5, -0.75), 'PPO6h': (0.5, -0.75),
        'PPO9h': (-0.7, -0.8), 'PPO10h': (0.7, -0.8), 'POO3h': (-0.3, -0.9), 'POO4h': (0.3, -0.9),
        'POO9h': (-0.6, -0.85), 'POO10h': (0.6, -0.85), 'OI1h': (-0.2, -0.95), 'OI2h': (0.2, -0.95),
        'I1': (-0.1, -0.95), 'Iz': (0.0, -0.95), 'I2': (0.1, -0.95),
        'AFp3h': (-0.35, 0.9), 'AFp4h': (0.35, 0.9)
    }
    
    # 为所有128个电极分配位置
    positions = []
    electrode_names = []
    
    for i, electrode in enumerate(real_electrodes):
        # 移除'EEG '前缀
        clean_name = electrode.replace('EEG ', '')
        electrode_names.append(clean_name)
        
        if clean_name in position_map:
            positions.append(position_map[clean_name])
        else:
            # 对于未知电极，基于命名规律估计位置
            # 这里使用简单的散布策略
            angle = (i / 128) * 2 * np.pi
            radius = 0.7 + 0.2 * np.random.random()
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))
    
    return electrode_names[:128], np.array(positions[:128])

def create_hgd_128_adjacency_matrix(method='spatial_functional'):
    """
    为HGD 128通道创建真实的邻接矩阵
    """
    
    electrode_names, positions = get_real_hgd_electrode_positions()
    n_electrodes = len(electrode_names)
    
    if method == 'spatial_distance':
        # 基于真实电极空间距离
        adjacency = np.zeros((n_electrodes, n_electrodes))
        
        for i in range(n_electrodes):
            for j in range(n_electrodes):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    # 邻近电极连接（距离阈值0.25）
                    if dist < 0.25:
                        adjacency[i][j] = np.exp(-(dist**2) / (2 * 0.1**2))
        
        # 自连接
        adjacency += np.eye(n_electrodes)
        
    elif method == 'functional_connectivity':
        # 基于运动执行任务的功能连接模式
        adjacency = np.eye(n_electrodes) * 0.5
        
        # 定义功能区域
        motor_cortex = []      # 主要运动皮层
        sensorimotor = []      # 感觉运动区
        frontal = []           # 额叶
        parietal = []          # 顶叶
        
        for i, name in enumerate(electrode_names):
            name_upper = name.upper()
            # 运动皮层 (C区域)
            if any(region in name_upper for region in ['C1', 'C2', 'C3', 'C4', 'CZ']):
                motor_cortex.append(i)
            # 感觉运动区 (FC, CP区域)
            elif any(region in name_upper for region in ['FC', 'CP']):
                sensorimotor.append(i)
            # 额叶 (F区域)
            elif any(region in name_upper for region in ['F', 'AF']):
                frontal.append(i)
            # 顶叶 (P区域)
            elif any(region in name_upper for region in ['P', 'PO']):
                parietal.append(i)
        
        # 设置功能连接强度
        # 运动皮层内部强连接
        for i in motor_cortex:
            for j in motor_cortex:
                if i != j:
                    adjacency[i][j] = 0.9
        
        # 运动皮层与感觉运动区
        for i in motor_cortex:
            for j in sensorimotor:
                adjacency[i][j] = adjacency[j][i] = 0.7
        
        # 感觉运动区内部
        for i in sensorimotor:
            for j in sensorimotor:
                if i != j:
                    adjacency[i][j] = 0.6
        
        # 长距离连接
        for i in motor_cortex + sensorimotor:
            for j in frontal + parietal:
                adjacency[i][j] = adjacency[j][i] = 0.3
    
    elif method == 'spatial_functional':
        # 混合方法：结合空间和功能连接
        # 先计算空间邻接矩阵
        spatial_adjacency = np.zeros((n_electrodes, n_electrodes))
        for i in range(n_electrodes):
            for j in range(n_electrodes):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < 0.25:
                        spatial_adjacency[i][j] = np.exp(-(dist**2) / (2 * 0.1**2))
        spatial_adjacency += np.eye(n_electrodes)
        
        # 再计算功能邻接矩阵
        functional_adjacency = np.eye(n_electrodes) * 0.5
        motor_cortex = []
        sensorimotor = []
        frontal = []
        parietal = []
        
        for i, name in enumerate(electrode_names):
            name_upper = name.upper()
            if any(region in name_upper for region in ['C1', 'C2', 'C3', 'C4', 'CZ']):
                motor_cortex.append(i)
            elif any(region in name_upper for region in ['FC', 'CP']):
                sensorimotor.append(i)
            elif any(region in name_upper for region in ['F', 'AF']):
                frontal.append(i)
            elif any(region in name_upper for region in ['P', 'PO']):
                parietal.append(i)
        
        # 设置功能连接
        for i in motor_cortex:
            for j in motor_cortex:
                if i != j:
                    functional_adjacency[i][j] = 0.9
        for i in motor_cortex:
            for j in sensorimotor:
                functional_adjacency[i][j] = functional_adjacency[j][i] = 0.7
        
        # 加权组合
        adjacency = 0.4 * spatial_adjacency + 0.6 * functional_adjacency
    
    # 行归一化
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    adjacency_normalized = adjacency / row_sums
    
    return adjacency_normalized.astype(np.float32), electrode_names

class GraphConvolutionLayer(Layer):
    """图卷积层，专门用于EEG电极拓扑"""
    
    def __init__(self, units, adjacency_matrix, activation='relu', dropout_rate=0.3, **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.units = units
        self.adjacency_matrix = tf.constant(adjacency_matrix, dtype=tf.float32)
        self.activation = activation
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # 权重矩阵
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
        
        super(GraphConvolutionLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # inputs: (batch_size, n_electrodes, features)
        
        # 特征变换: X * W
        support = tf.matmul(inputs, self.kernel)
        
        # 图卷积: A * X * W
        output = tf.matmul(self.adjacency_matrix, support)
        
        # 添加偏置
        output = tf.nn.bias_add(output, self.bias)
        
        # 激活函数
        if self.activation == 'relu':
            output = tf.nn.relu(output)
        elif self.activation == 'elu':
            output = tf.nn.elu(output)
        
        return output

def create_gcn_branch_hgd(input_layer, adjacency_matrix=None, gcn_units=[64, 32, 16], 
                         dropout_rate=0.3, name_prefix='hgd_gcn'):
    """
    为HGD 128通道数据创建GCN分支
    
    Parameters:
    -----------
    input_layer : tensor
        输入张量 (batch, 1, 128, 1000)
    adjacency_matrix : numpy.ndarray
        128x128 邻接矩阵
    gcn_units : list
        每层GCN的单元数
    dropout_rate : float
        Dropout率
    """
    
    if adjacency_matrix is None:
        adjacency_matrix, _ = create_hgd_128_adjacency_matrix('spatial_functional')
    
    # 输入形状转换: (batch, 1, 128, 1000) → (batch, 128, 1000)
    x = Lambda(lambda x: tf.squeeze(x, axis=1), name=f'{name_prefix}_squeeze')(input_layer)
    
    # 时间维度特征提取: (batch, 128, 1000) → (batch, 128, features)
    # 使用多种时间统计特征
    x_mean = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True), 
                   name=f'{name_prefix}_mean')(x)
    x_std = Lambda(lambda x: tf.math.reduce_std(x, axis=-1, keepdims=True), 
                  name=f'{name_prefix}_std')(x)
    x_max = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True), 
                  name=f'{name_prefix}_max')(x)
    
    # 拼接时间特征: (batch, 128, 3)
    x_temporal = Concatenate(axis=-1, name=f'{name_prefix}_temporal_concat')([x_mean, x_std, x_max])
    
    # 应用多层GCN
    x_gcn = x_temporal
    for i, units in enumerate(gcn_units):
        x_gcn = GraphConvolutionLayer(
            units=units,
            adjacency_matrix=adjacency_matrix,
            activation='relu',
            name=f'{name_prefix}_gcn_{i+1}'
        )(x_gcn)
        
        x_gcn = BatchNormalization(name=f'{name_prefix}_bn_{i+1}')(x_gcn)
        x_gcn = Dropout(dropout_rate, name=f'{name_prefix}_dropout_{i+1}')(x_gcn)
    
    # 全局图池化: (batch, 128, features) → (batch, features)
    # 使用注意力加权池化
    attention_weights = Dense(1, activation='sigmoid', name=f'{name_prefix}_attention')(x_gcn)
    x_pooled = Lambda(
        lambda inputs: tf.reduce_sum(inputs[0] * inputs[1], axis=1),
        name=f'{name_prefix}_attention_pool'
    )([x_gcn, attention_weights])
    
    return x_pooled

def test_hgd_gcn():
    """测试HGD GCN实现"""
    
    print("=== 测试HGD 128通道GCN ===")
    
    # 创建邻接矩阵
    adj_matrix, electrode_names = create_hgd_128_adjacency_matrix('spatial_functional')
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print(f"电极名称示例: {electrode_names[:10]}")
    print(f"连接密度: {(adj_matrix > 0.1).sum() / (128*128):.3f}")
    
    # 测试GCN层
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    
    input_test = Input(shape=(1, 128, 1000))
    gcn_output = create_gcn_branch_hgd(input_test, adj_matrix)
    
    model_test = Model(inputs=input_test, outputs=gcn_output)
    print(f"GCN分支输出形状: {model_test.output_shape}")
    print(f"GCN分支参数量: {model_test.count_params()}")
    
    # 测试前向传播
    import numpy as np
    dummy_input = np.random.randn(4, 1, 128, 1000).astype(np.float32)
    output = model_test.predict(dummy_input, verbose=0)
    print(f"前向传播成功: {output.shape}")
    
    return adj_matrix, electrode_names

if __name__ == "__main__":
    test_hgd_gcn()
