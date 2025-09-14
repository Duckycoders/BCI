#!/usr/bin/env python3
"""
带进度条的HGD数据集下载脚本
"""

import os
import requests
import time
import sys
from tqdm import tqdm

def download_with_progress_bar(url, local_path, timeout=300):
    """带进度条的文件下载"""
    try:
        # 创建目录
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 检查现有文件
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            if size > 100 * 1024 * 1024:  # 大于100MB
                print(f"✓ 文件已存在: {os.path.basename(local_path)} ({size / (1024*1024):.1f} MB)")
                return True
            else:
                print(f"⚠️ 删除小文件: {size} 字节")
                os.remove(local_path)
        
        print(f"\n📥 下载: {os.path.basename(local_path)}")
        print(f"🔗 URL: {url}")
        
        # 获取文件大小
        response = requests.head(url, timeout=10)
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0:
            print("⚠️ 无法获取文件大小")
            return False
        
        print(f"📊 文件大小: {total_size / (1024*1024):.1f} MB")
        
        # 开始下载
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # 创建进度条
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"下载 {os.path.basename(local_path)}",
            ncols=80
        )
        
        downloaded = 0
        start_time = time.time()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress_bar.update(len(chunk))
                    
                    # 每秒更新一次速度信息
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed = downloaded / (1024 * 1024) / elapsed
                        progress_bar.set_postfix({
                            'speed': f'{speed:.1f} MB/s',
                            'ETA': f'{(total_size - downloaded) / (downloaded / elapsed) / 60:.1f}min' if downloaded > 0 else 'N/A'
                        })
        
        progress_bar.close()
        
        # 验证下载
        final_size = os.path.getsize(local_path)
        if final_size == total_size:
            elapsed_time = time.time() - start_time
            avg_speed = final_size / (1024 * 1024) / elapsed_time
            print(f"✅ 下载完成! 用时: {elapsed_time/60:.1f}分钟, 平均速度: {avg_speed:.1f} MB/s")
            return True
        else:
            print(f"❌ 文件大小不匹配: {final_size} vs {total_size}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⏸️ 下载被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

def download_hgd_subjects(subject_list=[1, 2, 3]):
    """下载指定受试者的HGD数据"""
    
    base_url = "https://gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data/"
    target_base = "C:/Users/徐善若/mne_data/high-gamma-dataset/data/"
    
    print("🚀 开始下载HGD数据集")
    print(f"📁 目标路径: {target_base}")
    print(f"👥 受试者: {subject_list}")
    print("=" * 60)
    
    success_count = 0
    total_files = len(subject_list) * 2  # 每个受试者有train和test
    
    for i, subject in enumerate(subject_list):
        print(f"\n📊 进度: 受试者 {subject} ({i+1}/{len(subject_list)})")
        print("-" * 40)
        
        subject_success = 0
        
        for data_type in ['train', 'test']:
            edf_url = f"{base_url}{data_type}/{subject}.edf"
            edf_path = os.path.join(target_base, data_type, f"{subject}.edf")
            
            if download_with_progress_bar(edf_url, edf_path):
                success_count += 1
                subject_success += 1
        
        print(f"📈 受试者 {subject} 完成: {subject_success}/2 个文件")
        
        # 短暂休息避免服务器限制
        if i < len(subject_list) - 1:  # 不是最后一个
            print("⏳ 等待3秒...")
            time.sleep(3)
    
    print("\n" + "=" * 60)
    print(f"🎯 总结: 成功下载 {success_count}/{total_files} 个文件")
    
    if success_count >= len(subject_list):  # 至少每个受试者有一个文件
        print("✅ 下载成功！可以开始训练了")
        return True
    else:
        print("❌ 下载不完整，请检查网络连接")
        return False

def quick_test_loading():
    """快速测试数据加载"""
    print("\n🧪 快速测试数据加载...")
    
    try:
        from preprocess_HGD import load_HGD_data_moabb
        
        data_path = "C:/Users/徐善若/mne_data/high-gamma-dataset/data/"
        X_train, y_train = load_HGD_data_moabb(data_path, 1, True)
        
        if X_train is not None:
            print(f"✅ 数据加载成功: {X_train.shape}")
            return True
        else:
            print("❌ 数据加载返回None")
            return False
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

if __name__ == "__main__":
    print("🎯 HGD数据集智能下载器")
    print("=" * 50)
    
    try:
        # 下载前3个受试者的数据
        download_success = download_hgd_subjects([1, 2, 3])
        
        if download_success:
            # 测试加载
            load_success = quick_test_loading()
            
            if load_success:
                print("\n🎉 HGD数据集准备完成！")
                print("💡 提示: 可以运行 main_TrainValTest.py 开始训练")
                print("📝 如需更多受试者数据，可修改 subject_list=[1,2,3,4,5...]")
            else:
                print("\n⚠️ 数据下载成功但加载有问题")
        else:
            print("\n❌ 数据下载失败")
            
    except KeyboardInterrupt:
        print("\n⏹️ 下载被用户中断")
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
