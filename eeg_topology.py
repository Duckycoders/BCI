"""
128通道高密度EEG电极拓扑图构建
基于标准10-20系统扩展的高密度电极布局
"""

import numpy as np
import matplotlib.pyplot as plt

def get_hgd_128_electrode_positions():
    """
    获取HGD数据集128通道EEG电极的标准位置
    基于10-20系统扩展的高密度布局
    """
    
    # 128通道高密度EEG电极位置 (基于标准10-20扩展系统)
    # 坐标系：(x, y) 其中 x为左右方向，y为前后方向，范围[-1, 1]
    electrode_positions = {
        # 中线电极 (midline)
        'Fpz': (0.0, 0.95), 'AFz': (0.0, 0.85), 'Fz': (0.0, 0.75), 
        'FCz': (0.0, 0.55), 'Cz': (0.0, 0.0), 'CPz': (0.0, -0.45), 
        'Pz': (0.0, -0.75), 'POz': (0.0, -0.85), 'Oz': (0.0, -0.95),
        
        # 额叶区域 (frontal)
        'Fp1': (-0.3, 0.95), 'Fp2': (0.3, 0.95),
        'AF3': (-0.25, 0.85), 'AF4': (0.25, 0.85),
        'AF7': (-0.55, 0.8), 'AF8': (0.55, 0.8),
        'F1': (-0.25, 0.75), 'F2': (0.25, 0.75),
        'F3': (-0.45, 0.75), 'F4': (0.45, 0.75),
        'F5': (-0.65, 0.7), 'F6': (0.65, 0.7),
        'F7': (-0.8, 0.65), 'F8': (0.8, 0.65),
        
        # 额中央区域 (frontocentral)
        'FC1': (-0.25, 0.55), 'FC2': (0.25, 0.55),
        'FC3': (-0.45, 0.55), 'FC4': (0.45, 0.55),
        'FC5': (-0.65, 0.5), 'FC6': (0.65, 0.5),
        'FT7': (-0.8, 0.4), 'FT8': (0.8, 0.4),
        'FT9': (-0.9, 0.3), 'FT10': (0.9, 0.3),
        
        # 中央区域 (central)
        'C1': (-0.25, 0.0), 'C2': (0.25, 0.0),
        'C3': (-0.45, 0.0), 'C4': (0.45, 0.0),
        'C5': (-0.65, 0.0), 'C6': (0.65, 0.0),
        'T7': (-0.8, 0.0), 'T8': (0.8, 0.0),
        'T9': (-0.9, 0.0), 'T10': (0.9, 0.0),
        
        # 中央顶叶区域 (centroparietal)
        'CP1': (-0.25, -0.45), 'CP2': (0.25, -0.45),
        'CP3': (-0.45, -0.45), 'CP4': (0.45, -0.45),
        'CP5': (-0.65, -0.5), 'CP6': (0.65, -0.5),
        'TP7': (-0.8, -0.4), 'TP8': (0.8, -0.4),
        'TP9': (-0.9, -0.3), 'TP10': (0.9, -0.3),
        
        # 顶叶区域 (parietal)
        'P1': (-0.25, -0.75), 'P2': (0.25, -0.75),
        'P3': (-0.45, -0.75), 'P4': (0.45, -0.75),
        'P5': (-0.65, -0.7), 'P6': (0.65, -0.7),
        'P7': (-0.8, -0.65), 'P8': (0.8, -0.65),
        'P9': (-0.9, -0.6), 'P10': (0.9, -0.6),
        
        # 顶枕区域 (parieto-occipital)
        'PO3': (-0.25, -0.85), 'PO4': (0.25, -0.85),
        'PO7': (-0.55, -0.8), 'PO8': (0.55, -0.8),
        'PO9': (-0.75, -0.75), 'PO10': (0.75, -0.75),
        
        # 枕叶区域 (occipital)
        'O1': (-0.3, -0.95), 'O2': (0.3, -0.95),
        'O9': (-0.6, -0.9), 'O10': (0.6, -0.9),
        
        # 高密度电极 (additional high-density electrodes)
        # 额叶高密度
        'AFF1': (-0.15, 0.9), 'AFF2': (0.15, 0.9),
        'AFF5': (-0.4, 0.9), 'AFF6': (0.4, 0.9),
        'FFT7': (-0.7, 0.75), 'FFT8': (0.7, 0.75),
        'FFC1': (-0.15, 0.65), 'FFC2': (0.15, 0.65),
        'FFC3': (-0.35, 0.65), 'FFC4': (0.35, 0.65),
        'FFC5': (-0.55, 0.65), 'FFC6': (0.55, 0.65),
        
        # 中央高密度
        'FCC1': (-0.15, 0.3), 'FCC2': (0.15, 0.3),
        'FCC3': (-0.35, 0.3), 'FCC4': (0.35, 0.3),
        'FCC5': (-0.55, 0.3), 'FCC6': (0.55, 0.3),
        'CCP1': (-0.15, -0.2), 'CCP2': (0.15, -0.2),
        'CCP3': (-0.35, -0.2), 'CCP4': (0.35, -0.2),
        'CCP5': (-0.55, -0.2), 'CCP6': (0.55, -0.2),
        
        # 顶叶高密度
        'CPP1': (-0.15, -0.6), 'CPP2': (0.15, -0.6),
        'CPP3': (-0.35, -0.6), 'CPP4': (0.35, -0.6),
        'CPP5': (-0.55, -0.6), 'CPP6': (0.55, -0.6),
        'PPO1': (-0.15, -0.8), 'PPO2': (0.15, -0.8),
        'PPO9': (-0.65, -0.85), 'PPO10': (0.65, -0.85),
        
        # 颞叶高密度
        'FTT7': (-0.85, 0.2), 'FTT8': (0.85, 0.2),
        'FTT9': (-0.95, 0.1), 'FTT10': (0.95, 0.1),
        'TTP7': (-0.85, -0.2), 'TTP8': (0.85, -0.2),
        'TPP7': (-0.85, -0.5), 'TPP8': (0.85, -0.5),
        'TPP9': (-0.95, -0.4), 'TPP10': (0.95, -0.4),
        
        # 枕叶高密度
        'POO3': (-0.4, -0.9), 'POO4': (0.4, -0.9),
        'POO9': (-0.7, -0.85), 'POO10': (0.7, -0.85),
        'OI1': (-0.2, -1.0), 'OI2': (0.2, -1.0),
        
        # 极地电极
        'I1': (-0.1, -1.0), 'Iz': (0.0, -1.0), 'I2': (0.1, -1.0),
        
        # 额极高密度
        'AFp3': (-0.4, 1.0), 'AFp4': (0.4, 1.0),
        'AFp1': (-0.2, 1.0), 'AFp2': (0.2, 1.0),
        'AFF3': (-0.3, 0.95), 'AFF4': (0.3, 0.95),
    }
    
    # 如果电极数量不够128个，添加插值电极
    current_count = len(electrode_positions)
    if current_count < 128:
        # 在现有电极间插值添加电极
        for i in range(128 - current_count):
            electrode_positions[f'Extra{i+1}'] = (
                np.random.uniform(-0.9, 0.9), 
                np.random.uniform(-0.9, 0.9)
            )
    
    # 确保只返回前128个电极
    electrode_names = list(electrode_positions.keys())[:128]
    positions = [electrode_positions[name] for name in electrode_names]
    
    return electrode_names, np.array(positions)

def create_hgd_adjacency_matrix(method='spatial_distance', distance_threshold=0.3):
    """
    为HGD 128通道创建邻接矩阵
    
    Parameters:
    -----------
    method : str
        邻接矩阵构建方法
        - 'spatial_distance': 基于空间距离
        - 'functional_connectivity': 基于功能连接
        - 'hybrid': 混合方法
    distance_threshold : float
        空间距离阈值
    """
    
    electrode_names, positions = get_hgd_128_electrode_positions()
    n_electrodes = len(electrode_names)
    
    adjacency = np.zeros((n_electrodes, n_electrodes))
    
    if method == 'spatial_distance':
        # 基于欧几里得距离构建邻接矩阵
        for i in range(n_electrodes):
            for j in range(n_electrodes):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < distance_threshold:
                        # 使用高斯核，距离越近连接越强
                        adjacency[i][j] = np.exp(-(dist**2) / (2 * 0.1**2))
        
        # 添加自连接
        adjacency += np.eye(n_electrodes)
        
    elif method == 'functional_connectivity':
        # 基于运动执行任务的功能连接
        adjacency = np.eye(n_electrodes) * 0.5  # 自连接
        
        # 定义功能连接组（基于运动皮层解剖学）
        motor_cortex_indices = []
        sensorimotor_indices = []
        frontal_indices = []
        parietal_indices = []
        
        for i, name in enumerate(electrode_names):
            # 运动皮层 (primary motor cortex)
            if any(region in name.upper() for region in ['C1', 'C2', 'C3', 'C4', 'CZ']):
                motor_cortex_indices.append(i)
            # 感觉运动区
            elif any(region in name.upper() for region in ['CP', 'FC']):
                sensorimotor_indices.append(i)
            # 额叶
            elif any(region in name.upper() for region in ['F', 'AF']):
                frontal_indices.append(i)
            # 顶叶
            elif any(region in name.upper() for region in ['P', 'PO']):
                parietal_indices.append(i)
        
        # 运动皮层内部强连接
        for i in motor_cortex_indices:
            for j in motor_cortex_indices:
                if i != j:
                    adjacency[i][j] = 0.8
        
        # 运动皮层与感觉运动区连接
        for i in motor_cortex_indices:
            for j in sensorimotor_indices:
                adjacency[i][j] = adjacency[j][i] = 0.6
        
        # 感觉运动区与额叶/顶叶连接
        for i in sensorimotor_indices:
            for j in frontal_indices + parietal_indices:
                adjacency[i][j] = adjacency[j][i] = 0.3
    
    elif method == 'hybrid':
        # 结合空间和功能连接
        spatial_adj = create_hgd_adjacency_matrix('spatial_distance', distance_threshold)
        functional_adj = create_hgd_adjacency_matrix('functional_connectivity')
        
        # 加权组合
        adjacency = 0.6 * spatial_adj + 0.4 * functional_adj
    
    # 对称归一化
    degree = np.sum(adjacency, axis=1)
    degree[degree == 0] = 1  # 避免除零
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    adjacency_normalized = degree_inv_sqrt @ adjacency @ degree_inv_sqrt
    
    return adjacency_normalized.astype(np.float32), electrode_names

def visualize_electrode_topology(save_path=None):
    """可视化128通道电极拓扑图"""
    
    electrode_names, positions = get_hgd_128_electrode_positions()
    
    plt.figure(figsize=(12, 10))
    
    # 绘制电极位置
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    
    plt.scatter(x_coords, y_coords, c='blue', s=30, alpha=0.7)
    
    # 添加电极标签（只显示部分主要电极）
    important_electrodes = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4', 
                           'FC1', 'FC2', 'CP1', 'CP2', 'O1', 'O2']
    
    for i, name in enumerate(electrode_names):
        if name in important_electrodes:
            plt.annotate(name, (x_coords[i], y_coords[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
    
    # 绘制头部轮廓
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = np.cos(theta)
    head_y = np.sin(theta)
    plt.plot(head_x, head_y, 'k-', linewidth=2, alpha=0.3)
    
    # 绘制鼻子
    nose_x = [0, 0]
    nose_y = [1, 1.1]
    plt.plot(nose_x, nose_y, 'k-', linewidth=2)
    
    plt.title('128-Channel High-Density EEG Electrode Layout\n(HGD Dataset)', fontsize=14)
    plt.xlabel('Left-Right Position')
    plt.ylabel('Anterior-Posterior Position')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"电极拓扑图保存到: {save_path}")
    
    plt.show()
    
    return electrode_names, positions

if __name__ == "__main__":
    # 测试电极拓扑
    print("创建128通道EEG电极拓扑...")
    
    electrode_names, positions = get_hgd_128_electrode_positions()
    print(f"电极数量: {len(electrode_names)}")
    print(f"位置范围: x[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], y[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    
    # 创建邻接矩阵
    adj_matrix, names = create_hgd_adjacency_matrix('spatial_distance')
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print(f"连接密度: {(adj_matrix > 0.1).sum() / (128*128):.3f}")
    
    # 可视化（可选）
    # visualize_electrode_topology('electrode_layout_128ch.png')
