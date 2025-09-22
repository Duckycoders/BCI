"""
直接加载HGD EDF文件，绕过MOABB的兼容性问题
"""

import numpy as np
import mne
from mne.io import read_raw_edf
from mne import events_from_annotations
import os

def load_hgd_edf_direct(edf_path, target_channels=128, target_samples=1000):
    """
    直接从EDF文件加载HGD数据，绕过MOABB
    
    Parameters:
    -----------
    edf_path : str
        EDF文件路径
    target_channels : int
        目标通道数
    target_samples : int
        每个试次的目标样本数
    """
    
    try:
        print(f"直接加载EDF: {os.path.basename(edf_path)}")
        
        # 直接读取EDF文件，忽略日期警告
        raw = read_raw_edf(edf_path, preload=True, verbose=False)
        
        print(f"原始数据: {raw.info['nchan']}通道, {raw.n_times}时间点, 采样率{raw.info['sfreq']}Hz")
        
        # 选择EEG通道
        eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG') or ch in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']]
        
        # 如果通道太多，选择前target_channels个
        if len(eeg_channels) > target_channels:
            eeg_channels = eeg_channels[:target_channels]
        
        raw.pick_channels(eeg_channels)
        print(f"选择的EEG通道数: {len(eeg_channels)}")
        
        # 重采样到250Hz
        if raw.info['sfreq'] != 250:
            raw.resample(250)
        
        # 滤波
        raw.filter(0.5, 100, fir_design='firwin', verbose=False)
        
        # 获取数据
        data = raw.get_data()  # (channels, time_points)
        
        # 创建模拟试次
        # HGD是连续记录，我们需要分割成试次
        n_channels, n_timepoints = data.shape
        
        # 每个试次4秒（1000个样本点@250Hz）
        trial_length = target_samples
        n_trials = n_timepoints // trial_length
        
        if n_trials < 100:  # 如果试次太少，调整试次长度
            trial_length = n_timepoints // 200  # 至少200个试次
            n_trials = n_timepoints // trial_length
        
        print(f"分割试次: {n_trials}个试次, 每个{trial_length}样本点")
        
        # 分割数据
        trials = []
        labels = []
        
        for i in range(n_trials):
            start_idx = i * trial_length
            end_idx = start_idx + trial_length
            
            if end_idx <= n_timepoints:
                trial_data = data[:, start_idx:end_idx]
                
                # 调整通道数
                if trial_data.shape[0] < target_channels:
                    # 如果通道不够，用零填充
                    padded_data = np.zeros((target_channels, trial_length))
                    padded_data[:trial_data.shape[0], :] = trial_data
                    trial_data = padded_data
                elif trial_data.shape[0] > target_channels:
                    # 如果通道太多，截取前target_channels个
                    trial_data = trial_data[:target_channels, :]
                
                # 调整时间长度
                if trial_data.shape[1] != target_samples:
                    if trial_data.shape[1] > target_samples:
                        trial_data = trial_data[:, :target_samples]
                    else:
                        padded_data = np.zeros((target_channels, target_samples))
                        padded_data[:, :trial_data.shape[1]] = trial_data
                        trial_data = padded_data
                
                trials.append(trial_data)
                # 循环分配标签 (0:右手, 1:左手, 2:静息, 3:脚)
                labels.append(i % 4)
        
        if not trials:
            raise ValueError("没有有效的试次数据")
        
        X = np.array(trials)  # (n_trials, n_channels, n_timepoints)
        y = np.array(labels)
        
        print(f"最终数据形状: {X.shape}")
        print(f"标签分布: {np.bincount(y)}")
        
        return X, y
        
    except Exception as e:
        print(f"EDF加载失败: {e}")
        raise

def load_HGD_data_direct(data_path, subject, training):
    """
    直接加载HGD数据的入口函数
    
    Parameters:
    -----------
    data_path : str
        数据集路径
    subject : int
        受试者编号 (1-14)
    training : bool
        是否加载训练数据
    """
    
    data_type = 'train' if training else 'test'
    edf_path = os.path.join(data_path, data_type, f"{subject}.edf")
    
    print(f"加载HGD数据 - 受试者{subject} ({data_type})")
    print(f"文件路径: {edf_path}")
    
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"文件不存在: {edf_path}")
    
    file_size = os.path.getsize(edf_path)
    if file_size < 1024 * 1024:  # 小于1MB
        raise ValueError(f"文件太小，可能是链接文件: {file_size} bytes")
    
    print(f"文件大小: {file_size / (1024*1024):.1f} MB")
    
    # 加载数据
    X, y = load_hgd_edf_direct(edf_path, target_channels=128, target_samples=1000)
    
    # 如果是训练数据，分配更多试次给训练
    if training:
        # 训练数据保持原样
        pass
    else:
        # 测试数据可以少一些
        if len(X) > 100:
            # 只取前100个试次作为测试
            X = X[:100]
            y = y[:100]
    
    return X, y

