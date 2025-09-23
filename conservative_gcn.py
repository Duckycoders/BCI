"""
保守的GCN实现：只使用确定位置的标准10-20电极
避免对高密度电极位置的不准确估计
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, Lambda

def get_standard_electrode_mapping():
    """
    获取HGD数据集中标准10-20电极的确定位置
    只包含位置确定的电极，避免估计误差
    """
    
    # HGD数据集的标准10-20电极（位置确定）
    standard_electrodes = {
        'Fp1': (-0.309, 0.951), 'Fp2': (0.309, 0.951), 'Fpz': (0.0, 0.999),
        'F7': (-0.809, 0.588), 'F3': (-0.545, 0.839), 'Fz': (0.0, 0.719),
        'F4': (0.545, 0.839), 'F8': (0.809, 0.588),
        'FC5': (-0.719, 0.695), 'FC1': (-0.275, 0.719), 'FC2': (0.275, 0.719),
        'FC6': (0.719, 0.695), 'FCz': (0.0, 0.719),
        'T7': (-1.0, 0.0), 'C3': (-0.719, 0.695), 'Cz': (0.0, 0.0),
        'C4': (0.719, 0.695), 'T8': (1.0, 0.0),
        'CP5': (-0.719, -0.695), 'CP1': (-0.275, -0.719), 'CP2': (0.275, -0.719),
        'CP6': (0.719, -0.695), 'CPz': (0.0, -0.719),
        'P7': (-0.809, -0.588), 'P3': (-0.545, -0.839), 'Pz': (0.0, -0.719),
        'P4': (0.545, -0.839), 'P8': (0.809, -0.588),
        'PO7': (-0.695, -0.719), 'PO3': (-0.374, -0.927), 'POz': (0.0, -0.999),
        'PO4': (0.374, -0.927), 'PO8': (0.695, -0.719),
        'O1': (-0.309, -0.951), 'Oz': (0.0, -0.999), 'O2': (0.309, -0.951)
    }
    
    return standard_electrodes

def create_conservative_adjacency_matrix():
    """
    创建保守的邻接矩阵：只对确定位置的电极建立连接
    对其他电极使用功能连接
    """
    
    # 获取所有128个电极名称（从实际数据）
    from mne.io import read_raw_edf
    edf_path = 'C:/Users/徐善若/mne_data/high-gamma-dataset/data/train/1.edf'
    raw = read_raw_edf(edf_path, preload=False, verbose=False)
    all_electrode_names = [ch.replace('EEG ', '') for ch in raw.ch_names if ch.startswith('EEG')][:128]
    
    standard_electrodes = get_standard_electrode_mapping()
    n_electrodes = len(all_electrode_names)
    
    adjacency = np.eye(n_electrodes) * 0.5  # 自连接
    
    # 1. 对标准电极使用空间连接
    standard_indices = {}
    for i, name in enumerate(all_electrode_names):
        if name in standard_electrodes:
            standard_indices[name] = i
    
    print(f'找到 {len(standard_indices)} 个标准电极')
    
    # 标准电极间的空间连接
    for name1, idx1 in standard_indices.items():
        pos1 = np.array(standard_electrodes[name1])
        for name2, idx2 in standard_indices.items():
            if name1 != name2:
                pos2 = np.array(standard_electrodes[name2])
                dist = np.linalg.norm(pos1 - pos2)
                if dist < 0.4:  # 邻近电极
                    adjacency[idx1][idx2] = np.exp(-(dist**2) / (2 * 0.15**2))
    
    # 2. 对所有电极使用功能连接
    motor_regions = []
    sensorimotor_regions = []
    frontal_regions = []
    parietal_regions = []
    
    for i, name in enumerate(all_electrode_names):
        name_upper = name.upper()
        # 运动区
        if any(region in name_upper for region in ['C1', 'C2', 'C3', 'C4', 'CZ']):
            motor_regions.append(i)
        # 感觉运动区
        elif any(region in name_upper for region in ['FC', 'CP']) and 'FCC' not in name_upper and 'CCP' not in name_upper:
            sensorimotor_regions.append(i)
        # 额叶
        elif name_upper.startswith('F') and not name_upper.startswith('FC'):
            frontal_regions.append(i)
        # 顶叶
        elif name_upper.startswith('P') and not name_upper.startswith('PO'):
            parietal_regions.append(i)
    
    print(f'运动区电极: {len(motor_regions)}个')
    print(f'感觉运动区: {len(sensorimotor_regions)}个')
    
    # 功能连接
    # 运动区内部强连接
    for i in motor_regions:
        for j in motor_regions:
            if i != j:
                adjacency[i][j] = max(adjacency[i][j], 0.8)
    
    # 运动区与感觉运动区连接
    for i in motor_regions:
        for j in sensorimotor_regions:
            adjacency[i][j] = adjacency[j][i] = max(adjacency[i][j], 0.6)
    
    # 行归一化
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adjacency_normalized = adjacency / row_sums
    
    return adjacency_normalized.astype(np.float32), all_electrode_names

if __name__ == '__main__':
    adj, names = create_conservative_adjacency_matrix()
    print(f'邻接矩阵形状: {adj.shape}')
    print(f'连接密度: {(adj > 0.1).sum() / (128*128):.3f}')
"
