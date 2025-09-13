#!/usr/bin/env python3
"""
测试HGD数据集的设置和下载
"""

import os
import sys

def check_braindecode():
    """检查braindecode库是否安装"""
    try:
        import braindecode
        print(f"✓ Braindecode已安装，版本: {braindecode.__version__}")
        return True
    except ImportError:
        print("✗ Braindecode未安装")
        print("请运行: pip install braindecode")
        return False

def download_hgd_data():
    """下载HGD数据集"""
    print("=== HGD数据集下载和设置 ===")
    
    if not check_braindecode():
        return False
    
    try:
        # 设置数据路径
        home_dir = os.path.expanduser('~')
        hgd_path = os.path.join(home_dir, 'mne_data', 'MNE-schirrmeister2017-data', 
                               'robintibor', 'high-gamma-dataset', 'raw', 'master', 'data')
        
        print(f"HGD数据路径: {hgd_path}")
        
        # 检查数据是否已存在
        if os.path.exists(hgd_path):
            print("✓ HGD数据路径已存在")
            
            # 检查是否有数据文件
            train_files = []
            test_files = []
            
            train_dir = os.path.join(hgd_path, 'train')
            test_dir = os.path.join(hgd_path, 'test')
            
            if os.path.exists(train_dir):
                train_files = [f for f in os.listdir(train_dir) if f.endswith('.mat')]
                print(f"训练文件数量: {len(train_files)}")
                
            if os.path.exists(test_dir):
                test_files = [f for f in os.listdir(test_dir) if f.endswith('.mat')]
                print(f"测试文件数量: {len(test_files)}")
                
            if len(train_files) > 0 and len(test_files) > 0:
                print("✓ HGD数据文件已存在")
                return True
            else:
                print("⚠️ HGD路径存在但数据文件不完整")
        
        # 尝试下载数据
        print("正在下载HGD数据集...")
        print("这可能需要一些时间...")
        
        # 使用braindecode自动下载
        from braindecode.datasets.moabb import MOABBDataset
        from moabb.datasets import Schirrmeister2017
        
        # 创建数据集对象（这会自动下载数据）
        dataset = Schirrmeister2017()
        
        # 获取第一个受试者的数据来触发下载
        from moabb.paradigms import MotorImagery
        paradigm = MotorImagery()
        
        print("正在获取数据（自动下载）...")
        X, labels, meta = paradigm.get_data(dataset, [1])  # 获取第一个受试者
        
        print(f"✓ 数据下载成功!")
        print(f"数据形状: {X.shape}")
        print(f"标签: {np.unique(labels)}")
        print(f"受试者数量: {len(dataset.subject_list)}")
        
        return True
        
    except Exception as e:
        print(f"✗ HGD数据集下载失败: {e}")
        print("\n手动下载方法:")
        print("1. 访问: https://gin.g-node.org/robintibor/high-gamma-dataset")
        print("2. 下载数据到: ~/mne_data/MNE-schirrmeister2017-data/robintibor/high-gamma-dataset/raw/master/data/")
        print("3. 确保有train/和test/子文件夹，包含.mat文件")
        return False

def test_hgd_loading():
    """测试HGD数据加载"""
    try:
        from preprocess_HGD import load_HGD_data
        
        # 设置数据路径
        home_dir = os.path.expanduser('~')
        data_path = os.path.join(home_dir, 'mne_data', 'MNE-schirrmeister2017-data', 
                                'robintibor', 'high-gamma-dataset', 'raw', 'master', 'data') + '/'
        
        print(f"\n=== 测试HGD数据加载 ===")
        print(f"数据路径: {data_path}")
        
        # 测试加载第一个受试者
        X_train, y_train = load_HGD_data(data_path, 1, True)
        X_test, y_test = load_HGD_data(data_path, 1, False)
        
        print(f"✓ HGD数据加载成功!")
        print(f"训练数据形状: {X_train.shape}")
        print(f"测试数据形状: {X_test.shape}")
        print(f"训练标签分布: {np.bincount(y_train)}")
        print(f"测试标签分布: {np.bincount(y_test)}")
        
        return True, data_path
        
    except Exception as e:
        print(f"✗ HGD数据加载失败: {e}")
        return False, None

if __name__ == "__main__":
    import numpy as np
    
    print("开始HGD数据集设置...")
    
    # 下载数据
    download_success = download_hgd_data()
    
    if download_success:
        # 测试加载
        load_success, data_path = test_hgd_loading()
        
        if load_success:
            print(f"\n🎉 HGD数据集设置完成!")
            print(f"数据路径: {data_path}")
            print("\n可以在main_TrainValTest.py中设置:")
            print(f"dataset = 'HGD'")
            print(f"data_path = '{data_path}'")
        else:
            print("\n❌ HGD数据加载测试失败")
    else:
        print("\n❌ HGD数据集下载失败")

