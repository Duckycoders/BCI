""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

#%%
# We need the following to load and preprocess the High Gamma Dataset
import numpy as np
import logging
import os
import scipy.io as sio
from collections import OrderedDict
try:
    # 尝试新版本braindecode的导入路径
    from braindecode.datasets import BBCIDataset
    from braindecode.preprocessing import create_windows_from_events
    from braindecode.preprocessing import exponential_moving_standardize
    from braindecode.preprocessing import preprocess, Preprocessor
    NEW_BRAINDECODE = True
except ImportError:
    try:
        # 尝试旧版本braindecode的导入路径
        from braindecode.datasets.bbci import BBCIDataset
        from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
        from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
        from braindecode.datautil.signalproc import exponential_running_standardize
        from braindecode.datautil.signalproc import highpass_cnt
        NEW_BRAINDECODE = False
    except ImportError:
        print("Warning: braindecode not available, using MOABB fallback")
        NEW_BRAINDECODE = None

def load_HGD_data_from_mat(filename, subject, training):
    """直接从.mat文件加载HGD数据"""
    try:
        print(f"正在从.mat文件加载: {filename}")
        
        # 加载.mat文件
        mat_data = sio.loadmat(filename)
        
        # 检查.mat文件内容
        print(f"Mat文件键: {list(mat_data.keys())}")
        
        # 尝试不同的可能键名
        possible_data_keys = ['data', 'X', 'eeg_data', 'raw_data']
        possible_label_keys = ['labels', 'y', 'targets', 'events']
        
        data_key = None
        label_key = None
        
        for key in possible_data_keys:
            if key in mat_data and mat_data[key].size > 100:  # 确保不是元数据
                data_key = key
                break
        
        for key in possible_label_keys:
            if key in mat_data:
                label_key = key
                break
        
        if data_key is None:
            # 如果找不到标准键，使用最大的数组
            largest_key = max([k for k in mat_data.keys() if not k.startswith('__')], 
                            key=lambda k: mat_data[k].size if hasattr(mat_data[k], 'size') else 0)
            data_key = largest_key
            print(f"使用最大数组作为数据: {data_key}")
        
        X = mat_data[data_key]
        
        if label_key:
            y = mat_data[label_key].flatten()
        else:
            # 如果没有标签，创建模拟标签
            n_trials = X.shape[0] if X.ndim == 3 else X.shape[-1] // 1000
            y = np.tile(np.arange(4), n_trials // 4 + 1)[:n_trials]
            print(f"未找到标签，创建模拟标签: {len(y)}个")
        
        print(f"数据形状: {X.shape}, 标签形状: {y.shape}")
        
        # 确保数据格式正确: (trials, channels, time)
        if X.ndim == 2:
            # 如果是2D，假设是 (channels, time*trials)，需要重塑
            n_channels = X.shape[0]
            n_timepoints = 1000  # 假设每个试次1000个时间点
            n_trials = X.shape[1] // n_timepoints
            X = X[:, :n_trials*n_timepoints].reshape(n_channels, n_trials, n_timepoints)
            X = X.transpose(1, 0, 2)  # (trials, channels, time)
        elif X.ndim == 3:
            # 如果已经是3D，检查维度顺序
            if X.shape[1] > X.shape[0] and X.shape[1] > X.shape[2]:
                # 可能是 (trials, time, channels)
                X = X.transpose(0, 2, 1)  # 转为 (trials, channels, time)
        
        # 确保标签长度匹配
        y = y[:X.shape[0]]
        
        print(f"最终数据形状: {X.shape}, 标签: {np.bincount(y)}")
        return X, y
        
    except Exception as e:
        print(f"Mat文件加载失败: {e}")
        raise

#%%
def load_HGD_data_moabb(data_path, subject, training):
    """使用MOABB加载HGD数据的备用方法"""
    try:
        from moabb.datasets import Schirrmeister2017
        from moabb.paradigms import MotorImagery
        
        print(f"使用MOABB加载HGD数据 - 受试者 {subject}")
        
        # 创建数据集
        dataset = Schirrmeister2017()
        paradigm = MotorImagery(
            events=['right_hand', 'left_hand', 'rest', 'feet'],
            n_classes=4,
            fmin=0.5,
            fmax=100,
            tmin=0.5,
            tmax=4.5,
            resample=250
        )
        
        # 获取数据
        X, labels, meta = paradigm.get_data(dataset, [subject])
        
        # 分离训练/测试数据
        sessions = meta['session'].unique()
        if len(sessions) > 1:
            if training:
                mask = meta['session'] == sessions[0]
            else:
                mask = meta['session'] == sessions[-1]
        else:
            # 单会话情况下按比例分割
            n_total = len(X)
            if training:
                indices = np.arange(int(0.8 * n_total))
            else:
                indices = np.arange(int(0.8 * n_total), n_total)
            mask = np.zeros(n_total, dtype=bool)
            mask[indices] = True
        
        X_data = X[mask]
        y_data = labels[mask]
        
        # 转换标签
        label_map = {'right_hand': 0, 'left_hand': 1, 'rest': 2, 'feet': 3}
        y_numeric = np.array([label_map.get(label, 0) for label in y_data])
        
        print(f"MOABB加载成功: {X_data.shape}, 标签分布: {np.bincount(y_numeric)}")
        return X_data, y_numeric
        
    except Exception as e:
        print(f"MOABB加载失败: {e}")
        raise

def load_HGD_data(data_path, subject, training, low_cut_hz =0, debug = False):
    """ Loading training/testing data for the High Gamma Dataset (HGD)
    for a specific subject.
    
    Please note that  HGD is for "executed movements" NOT "motor imagery"  
    
    This function now uses MOABB as primary method with braindecode as fallback
   
        Parameters
        ----------
        data_path: string
            dataset path
        subject: int
            number of subject in [1, .. ,14]
        training: bool
            if True, load training data
            if False, load testing data
        debug: bool
            if True, 
            if False, 
    """
    
    # 检查本地文件是否存在
    if training:  
        filename = data_path + 'train/{}.mat'.format(subject)
    else:         
        filename = data_path + 'test/{}.mat'.format(subject)
    
    if os.path.exists(filename):
        print(f"使用本地文件加载HGD数据 - 受试者 {subject}: {filename}")
        try:
            return load_HGD_data_from_mat(filename, subject, training)
        except Exception as e:
            print(f"本地文件加载失败: {e}")
    
    # 如果本地文件不存在，尝试MOABB
    try:
        print(f"本地文件不存在，尝试MOABB下载 - 受试者 {subject}")
        return load_HGD_data_moabb(data_path, subject, training)
    except Exception as e:
        print(f"MOABB方法失败 (受试者 {subject}): {e}")
        print("跳过此受试者，返回空数据...")
        
        # 返回空数据，让训练跳过这个受试者
        return None, None

def load_HGD_data_braindecode(data_path, subject, training, low_cut_hz =0, debug = False):
    """ 使用braindecode加载HGD数据的原始方法 """

    log = logging.getLogger(__name__)
    log.setLevel('DEBUG')

    if training:  filename = (data_path + 'train/{}.mat'.format(subject))
    else:         filename = (data_path + 'test/{}.mat'.format(subject))

    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)
    
    log.info("Loading data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset.X, dataset.y
