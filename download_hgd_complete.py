#!/usr/bin/env python3
"""
完整下载HGD数据集的脚本
"""

import os
import requests
from urllib.parse import urljoin
import time

def download_file(url, local_path, timeout=300):
    """下载单个文件"""
    try:
        print(f"下载: {url}")
        print(f"保存到: {local_path}")
        
        # 创建目录
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 下载文件
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r进度: {progress:.1f}% ({downloaded}/{total_size})", end='')
        
        print(f"\n✓ 下载完成: {local_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False

def download_hgd_dataset():
    """下载完整的HGD数据集"""
    
    base_url = "https://gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data/"
    
    # 目标路径
    target_base = "C:/Users/徐善若/mne_data/MNE-schirrmeister2017-data/robintibor/high-gamma-dataset/raw/master/data/"
    
    print("=== 开始下载HGD数据集 ===")
    print(f"目标路径: {target_base}")
    
    success_count = 0
    total_count = 0
    
    # 下载所有受试者的数据
    for subject in range(1, 15):  # 14个受试者
        for data_type in ['train', 'test']:
            
            # 下载.edf文件（主要数据文件）
            edf_url = urljoin(base_url, f"{data_type}/{subject}.edf")
            edf_path = os.path.join(target_base, data_type, f"{subject}.edf")
            
            print(f"\n--- 受试者 {subject} ({data_type}) ---")
            
            total_count += 1
            if download_file(edf_url, edf_path):
                success_count += 1
                
                # 检查文件大小
                file_size = os.path.getsize(edf_path)
                print(f"文件大小: {file_size / (1024*1024):.1f} MB")
                
                if file_size < 1024:  # 如果文件太小，可能是错误
                    print("⚠️ 警告: 文件大小异常小，可能下载不完整")
            
            # 添加延迟避免服务器限制
            time.sleep(1)
    
    print(f"\n=== 下载完成 ===")
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 所有文件下载成功！")
        return True
    else:
        print(f"⚠️ 有 {total_count - success_count} 个文件下载失败")
        return False

def verify_hgd_data():
    """验证下载的HGD数据"""
    
    target_base = "C:/Users/徐善若/mne_data/MNE-schirrmeister2017-data/robintibor/high-gamma-dataset/raw/master/data/"
    
    print("\n=== 验证HGD数据 ===")
    
    train_files = []
    test_files = []
    
    for subject in range(1, 15):
        train_file = os.path.join(target_base, "train", f"{subject}.edf")
        test_file = os.path.join(target_base, "test", f"{subject}.edf")
        
        if os.path.exists(train_file):
            size = os.path.getsize(train_file) / (1024*1024)
            train_files.append((subject, size))
            
        if os.path.exists(test_file):
            size = os.path.getsize(test_file) / (1024*1024)
            test_files.append((subject, size))
    
    print(f"训练文件: {len(train_files)}/14")
    print(f"测试文件: {len(test_files)}/14")
    
    if train_files:
        print("训练文件大小:")
        for subject, size in train_files[:5]:  # 显示前5个
            print(f"  受试者 {subject}: {size:.1f} MB")
    
    if test_files:
        print("测试文件大小:")
        for subject, size in test_files[:5]:  # 显示前5个
            print(f"  受试者 {subject}: {size:.1f} MB")
    
    # 检查是否有足够的数据进行训练
    if len(train_files) >= 5 and len(test_files) >= 5:
        print("✓ 有足够的数据可以开始训练")
        return True
    else:
        print("✗ 数据不足，需要更多下载")
        return False

if __name__ == "__main__":
    print("开始HGD数据集完整下载...")
    
    # 下载数据
    download_success = download_hgd_dataset()
    
    # 验证数据
    verify_success = verify_hgd_data()
    
    if download_success and verify_success:
        print("\n🎉 HGD数据集准备完成！可以开始训练了。")
    else:
        print("\n❌ 请检查网络连接或手动下载数据")
        print("备用下载地址:")
        print("- https://gin.g-node.org/robintibor/high-gamma-dataset")
        print("- https://github.com/robintibor/high-gamma-dataset")
