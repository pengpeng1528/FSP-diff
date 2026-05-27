import numpy as np
from PIL import Image
import os
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import glob
from collections import defaultdict


def extract_base_name(filename):
    """从文件名中提取基础名称（去掉扩展名和_processed后缀）"""
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.endswith('_processed'):
        return base[:-10]  # 去掉'_processed'
    return base


def find_matching_pairs(dir1, dir2):
    """在两个目录中查找匹配的文件对"""
    files1 = glob.glob(os.path.join(dir1, "*.mat"))
    files2 = glob.glob(os.path.join(dir2, "*.mat"))
    
    dict1 = {extract_base_name(f): f for f in files1}
    dict2 = {extract_base_name(f): f for f in files2}
    
    common_bases = set(dict1.keys()) & set(dict2.keys())
    pairs = [(dict1[base], dict2[base]) for base in common_bases]
    
    return pairs


def load_image_raw(path):
    """加载多种格式的图像，不进行归一化"""
    print(f"加载图像: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    
    try:
        if ext == '.mat':
            mat_data = sio.loadmat(path)
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            
            if not data_keys:
                raise ValueError("在.mat文件中未找到有效的数据变量")
            
            data_key = data_keys[0]
            print(f"选择变量: {data_key}，形状: {mat_data[data_key].shape}")
            
            img = mat_data[data_key].astype(np.float32)
            
        elif ext == '.npy':
            img = np.load(path).astype(np.float32)
            
        else:
            img = Image.open(path).convert('L')
            img = np.array(img).astype(np.float32)
        
        print(f"原始图像值范围: 最小值={img.min():.6f}, 最大值={img.max():.6f}")
        return img
        
    except Exception as e:
        print(f"加载图像失败: {path}")
        print(f"错误: {str(e)}")
        raise


def joint_normalization(img1, img2):
    """对两个图像进行联合归一化"""
    global_min = min(img1.min(), img2.min())
    global_max = max(img1.max(), img2.max())
    
    if global_max > global_min:
        img1_norm = (img1 - global_min) / (global_max - global_min)
        img2_norm = (img2 - global_min) / (global_max - global_min)
        
        normalization_params = {
            'global_min': global_min,
            'global_max': global_max,
            'range': global_max - global_min,
            'method': 'joint_minmax'
        }
        
        return img1_norm, img2_norm, normalization_params
    else:
        if global_max == global_min:
            img1_norm = np.ones_like(img1) * 0.5
            img2_norm = np.ones_like(img2) * 0.5
            
            normalization_params = {
                'global_min': global_min,
                'global_max': global_max,
                'range': 0,
                'method': 'constant_image'
            }
            
            return img1_norm, img2_norm, normalization_params


def adjust_images(img1, img2):
    """确保两个图像有相同的形状"""
    if img1.shape != img2.shape:
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_height, :min_width]
        img2 = img2[:min_height, :min_width]
    return img1, img2


def process_channel(channel_idx, clean_ch, noisy_ch):
    """处理单个通道"""
    clean_ch, noisy_ch = adjust_images(clean_ch, noisy_ch)
    clean_norm, noisy_norm, _ = joint_normalization(clean_ch, noisy_ch)
    
    psnr = compare_psnr(clean_norm, noisy_norm, data_range=1.0)
    ssim = compare_ssim(clean_norm, noisy_norm, data_range=1.0)
    
    return psnr, ssim


def process_image_pair(clean_path, noisy_path):
    """处理一对图像"""
    print(f"\n处理图像对: {clean_path} 和 {noisy_path}")
    
    clean_raw = load_image_raw(clean_path)
    noisy_raw = load_image_raw(noisy_path)
    
    # 处理单通道图像
    if clean_raw.ndim == 2 and noisy_raw.ndim == 2:
        clean_raw, noisy_raw = adjust_images(clean_raw, noisy_raw)
        clean_norm, noisy_norm, _ = joint_normalization(clean_raw, noisy_raw)
        
        psnr = compare_psnr(clean_norm, noisy_norm, data_range=1.0)
        ssim = compare_ssim(clean_norm, noisy_norm, data_range=1.0)
        
        return {
            'psnr': [psnr],
            'ssim': [ssim],
            'channels': 1,
            'clean': clean_path,
            'noisy': noisy_path
        }
    
    # 处理多通道图像
    elif clean_raw.ndim == 3 and noisy_raw.ndim == 3:
        if clean_raw.shape[0] != noisy_raw.shape[0]:
            raise ValueError(f"通道数不匹配: {clean_raw.shape[0]} vs {noisy_raw.shape[0]}")
        
        channel_psnrs = []
        channel_ssims = []
        
        for c in range(clean_raw.shape[0]):
            psnr, ssim = process_channel(c, clean_raw[c], noisy_raw[c])
            channel_psnrs.append(psnr)
            channel_ssims.append(ssim)
            print(f"通道 {c+1} - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        
        return {
            'psnr': channel_psnrs,
            'ssim': channel_ssims,
            'channels': clean_raw.shape[0],
            'clean': clean_path,
            'noisy': noisy_path
        }
    
    else:
        raise ValueError("图像维度不匹配或不受支持")


def calculate_channel_averages(all_results):
    """计算每个通道的平均值"""
    channel_stats = defaultdict(lambda: {'psnr_sum': 0, 'ssim_sum': 0, 'count': 0})
    
    for result in all_results:
        for c in range(len(result['psnr'])):
            channel_stats[c]['psnr_sum'] += result['psnr'][c]
            channel_stats[c]['ssim_sum'] += result['ssim'][c]
            channel_stats[c]['count'] += 1
    
    # 转换为有序字典并按通道排序
    channel_averages = {}
    for c in sorted(channel_stats.keys()):
        count = channel_stats[c]['count']
        channel_averages[c] = {
            'avg_psnr': channel_stats[c]['psnr_sum'] / count,
            'avg_ssim': channel_stats[c]['ssim_sum'] / count,
            'count': count
        }
    
    return channel_averages


def main():
    clean_dir = '/SSD_8T/pengpeng/DiffIR-master03/FSP-diff/test/data/cleanval/val'
    noisy_dir = '/SSD_8T/pengpeng/DiffIR-master03/FSP-diff/test/out/valdenoisy'
    
    # 日志输出目录与文件
    log_dir = '/SSD_8T/pengpeng/DiffIR-master03/FSP-diff/results/testlog'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'metrics.txt')
    
    pairs = find_matching_pairs(clean_dir, noisy_dir)
    
    if not pairs:
        print("未找到匹配的图像对！")
        return
    
    print(f"找到 {len(pairs)} 对匹配图像")
    
    all_results = []
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"匹配图像对数: {len(pairs)}\n")
        f.write("每个文件每个通道的度量 (PSNR dB, SSIM):\n")
        f.write("="*60 + "\n")
        
        for clean_path, noisy_path in pairs:
            try:
                result = process_image_pair(clean_path, noisy_path)
                all_results.append(result)
                
                base_name = extract_base_name(os.path.basename(clean_path))
                f.write(f"文件: {base_name}\n")
                for c in range(len(result['psnr'])):
                    f.write(f"  通道 {c+1}: PSNR={result['psnr'][c]:.2f} dB, SSIM={result['ssim'][c]:.4f}\n")
                f.write("-"*40 + "\n")
                
                print(f"\n结果: {os.path.basename(clean_path)} vs {os.path.basename(noisy_path)}")
                for c in range(len(result['psnr'])):
                    print(f"通道 {c+1} - PSNR: {result['psnr'][c]:.2f} dB, SSIM: {result['ssim'][c]:.4f}")
            
            except Exception as e:
                print(f"处理 {clean_path} 和 {noisy_path} 时出错: {str(e)}")
                continue
        
        if all_results:
            # 计算每个通道的平均值
            channel_averages = calculate_channel_averages(all_results)
            
            # 计算全局平均值
            global_avg_psnr = np.mean([np.mean(r['psnr']) for r in all_results])
            global_avg_ssim = np.mean([np.mean(r['ssim']) for r in all_results])
            
            f.write("\n" + "="*50 + "\n")
            f.write("通道平均结果:\n")
            f.write("="*50 + "\n")
            for c, stats in channel_averages.items():
                f.write(f"通道 {c+1} (共 {stats['count']} 个样本):\n")
                f.write(f"  平均PSNR: {stats['avg_psnr']:.2f} dB\n")
                f.write(f"  平均SSIM: {stats['avg_ssim']:.4f}\n")
                f.write("-"*40 + "\n")
            
            f.write("\n全局平均结果:\n")
            f.write("="*50 + "\n")
            f.write(f"图像对数: {len(all_results)}\n")
            f.write(f"全局平均PSNR: {global_avg_psnr:.2f} dB\n")
            f.write(f"全局平均SSIM: {global_avg_ssim:.4f}\n")
            f.write("="*50 + "\n")
    
    # 控制台也保留打印每通道平均与全局平均
    if all_results:
        channel_averages = calculate_channel_averages(all_results)
        global_avg_psnr = np.mean([np.mean(r['psnr']) for r in all_results])
        global_avg_ssim = np.mean([np.mean(r['ssim']) for r in all_results])
        
        print("\n" + "="*50)
        print("通道平均结果:")
        print("="*50)
        for c, stats in channel_averages.items():
            print(f"通道 {c+1} (共 {stats['count']} 个样本):")
            print(f"  平均PSNR: {stats['avg_psnr']:.2f} dB")
            print(f"  平均SSIM: {stats['avg_ssim']:.4f}")
            print("-"*40)
        
        print("\n全局平均结果:")
        print("="*50)
        print(f"图像对数: {len(all_results)}")
        print(f"全局平均PSNR: {global_avg_psnr:.2f} dB")
        print(f"全局平均SSIM: {global_avg_ssim:.4f}")
        print("="*50)


if __name__ == '__main__':
    main()