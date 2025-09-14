#!/usr/bin/env python3
"""
直接下载HGD数据集的.edf文件
"""

import os
import requests
import time
from urllib.parse import urljoin

def download_large_file(url, local_path, chunk_size=8192):
    """下载大文件，显示进度"""
    try:
        print(f"开始下载: {os.path.basename(local_path)}")
        
        # 创建目录
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 检查文件是否已存在且大小合理
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            if size > 100 * 1024 * 1024:  # 大于100MB
                print(f"✓ 文件已存在且大小正常: {size / (1024*1024):.1f} MB")
                return True
            else:
                print(f"⚠️ 现有文件太小({size}字节)，重新下载...")
                os.remove(local_path)
        
        # 下载文件
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            start_time = time.time()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 显示进度
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            elapsed = time.time() - start_time
                            speed = downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
                            print(f"\r  进度: {progress:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB) 速度: {speed:.1f} MB/s", end='')
        
        print(f"\n✓ 下载完成!")
        
        # 验证文件大小
        final_size = os.path.getsize(local_path)
        if final_size > 100 * 1024 * 1024:
            print(f"✓ 文件大小验证通过: {final_size / (1024*1024):.1f} MB")
            return True
        else:
            print(f"✗ 文件大小异常: {final_size} 字节")
            return False
            
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False

def download_essential_hgd_data():
    """下载必要的HGD数据（前几个受试者）"""
    
    base_url = "https://gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data/"
    target_base = "C:/Users/徐善若/mne_data/high-gamma-dataset/data/"
    
    print("=== 下载必要的HGD数据 ===")
    print("先下载前3个受试者的数据用于测试...")
    
    success_count = 0
    
    # 只下载前3个受试者的数据
    for subject in range(1, 4):  # 受试者1-3
        for data_type in ['train', 'test']:
            
            edf_url = urljoin(base_url, f"{data_type}/{subject}.edf")
            edf_path = os.path.join(target_base, data_type, f"{subject}.edf")
            
            print(f"\n--- 受试者 {subject} ({data_type}) ---")
            
            if download_large_file(edf_url, edf_path):
                success_count += 1
            
            # 短暂暂停避免服务器限制
            time.sleep(2)
    
    print(f"\n=== 下载完成 ===")
    print(f"成功下载: {success_count}/6 个文件")
    
    if success_count >= 4:  # 至少有2个受试者的完整数据
        print("✓ 有足够的数据可以开始测试训练")
        return True
    else:
        print("✗ 下载的数据不足")
        return False

def test_hgd_loading():
    """测试HGD数据加载"""
    try:
        from preprocess_HGD import load_HGD_data_moabb
        
        data_path = "C:/Users/徐善若/mne_data/high-gamma-dataset/data/"
        
        print(f"\n=== 测试HGD数据加载 ===")
        
        # 测试第一个受试者
        X_train, y_train = load_HGD_data_moabb(data_path, 1, True)
        X_test, y_test = load_HGD_data_moabb(data_path, 1, False)
        
        print(f"✓ 数据加载成功!")
        print(f"训练数据: {X_train.shape}")
        print(f"测试数据: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始HGD数据集下载...")
    
    # 下载必要数据
    download_success = download_essential_hgd_data()
    
    if download_success:
        # 测试加载
        load_success = test_hgd_loading()
        
        if load_success:
            print("\n🎉 HGD数据集设置完成！可以开始训练。")
        else:
            print("\n⚠️ 数据下载成功但加载有问题，可能需要调试")
    else:
        print("\n❌ 数据下载失败，请检查网络连接")
