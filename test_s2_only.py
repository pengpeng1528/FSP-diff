#!/usr/bin/env python3
"""
使用S2模型处理图像的代码
基于原始测试逻辑，不依赖S1模型
"""

import os
import torch
import numpy as np
import argparse
from torch.nn import functional as F
import cv2
from PIL import Image
import yaml
from basicsr.utils import img2tensor, tensor2img, imfrombytes
from basicsr.archs import build_network
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

# 添加FSP项目路径（内部包名仍为FSP）
import sys
sys.path.append('./FSP')

import FSP.archs
import FSP.data
import FSP.models

# 添加缺失的imfrommat函数
def imfrommat(path, float32=True):
    """从.mat文件读取图像"""
    try:
        import scipy.io as sio
        mat = sio.loadmat(path)
        # 尝试不同的键名
        if 'img' in mat:
            img = mat['img']
        elif 'data' in mat:
            img = mat['data']
        elif 'image' in mat:
            img = mat['image']
        else:
            # 获取第一个数组
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if keys:
                img = mat[keys[0]]
            else:
                raise ValueError(f"无法在.mat文件中找到图像数据: {path}")
        
        if float32:
            img = img.astype(np.float32)
        return img
    except ImportError:
        print("警告: 需要安装scipy来读取.mat文件")
        raise
    except Exception as e:
        print(f"读取.mat文件时出错: {e}")
        raise

@MODEL_REGISTRY.register()
class S2OnlyModel(SRModel):
    """仅使用S2模型的简化版本"""
    
    def __init__(self, opt):
        super(S2OnlyModel, self).__init__(opt)
        
    def test(self):
        """测试方法 - 直接使用S2模型"""
        window_size = self.opt['val'].get('window_size', 0)
        
        if window_size:
            lq, gt, mod_pad_h, mod_pad_w = self.pad_test(window_size)
        else:
            lq = self.lq
            gt = self.gt
            
        # 使用EMA模型或普通模型
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # 直接调用S2模型，不传入S1的IPR
                self.output = self.net_g_ema(lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                # 直接调用S2模型，不传入S1的IPR
                self.output = self.net_g(lq)
            self.net_g.train()
            
        # 移除填充
        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
    
    def pad_test(self, window_size):
        """滑动窗口填充"""
        scale = 1
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
            
        lq = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt = F.pad(self.gt, (0, mod_pad_w*scale, 0, mod_pad_h*scale), 'reflect')
        return lq, gt, mod_pad_h, mod_pad_w

def create_s2_only_config():
    """创建仅使用S2的配置"""
    config = {
        'name': 'test_S2Only',
        'model_type': 'S2OnlyModel',
        'scale': 1,
        'num_gpu': 1,  # 改为整数
        'manual_seed': 0,
        'is_train': False,  # 添加这个参数
        'dist': False,  # 添加分布式训练参数
        
        'datasets': {
            'val_1': {
                'name': 'Test',
                'type': 'DeblurPairedDataset',
                'dataroot_gt': './test_data/gt',
                'dataroot_lq': './test_data/lq',
                'io_backend': {'type': 'disk'}
            }
        },
        
        'network_g': {
            'type': 'FSPS2',
            'n_encoder_res': 5,
            'inp_channels': 3,
            'out_channels': 3,
            'dim': 48,
            'num_blocks': [6,5,5,4],  # 使用您修改的配置
            'num_refinement_blocks': 4,
            'heads': [1,2,4,8],
            'ffn_expansion_factor': 2,
            'bias': False,
            'LayerNorm_type': 'WithBias',
            'n_denoise_res': 1,
            'linear_start': 0.1,
            'linear_end': 0.99,
            'timesteps': 4
        },
        
        'path': {
            'pretrain_network_g': '/SSD_8T/pengpeng/DiffIR-master03/FSP-demotionblur/experiments/train_FSPS2/models/net_g_130000.pth',
            'param_key_g': 'params_ema',
            'strict_load_g': False
        },
        
        'val': {
            'save_img': True,
            'suffix': '~',
            'metrics': {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 0,
                    'test_y_channel': False
                }
            }
        }
    }
    return config

def process_single_image(model, image_path, output_path, device='cuda'):
    """处理单张图像 - 每个通道独立归一化，保存为mat文件"""
    print(f"正在处理图像: {image_path}")
    
    # 加载图像
    if image_path.endswith('.mat'):
        img = imfrommat(image_path, float32=True)
        # 检查图像维度
        if img.ndim == 3:
            # 如果是3维，假设是HWC格式
            if img.shape[2] == 3:  # 最后一维是通道
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> BCHW
            else:  # 第一维是通道
                img = torch.from_numpy(img).unsqueeze(0)  # CHW -> BCHW
        elif img.ndim == 2:
            # 如果是2维，转换为3通道
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # HW -> BHW -> BCHW
            img = img.repeat(1, 3, 1, 1)  # 复制到3通道
        else:
            img = torch.from_numpy(img).unsqueeze(0)
    else:
        img = Image.open(image_path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.
        img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0)
    
    img = img.to(device)
    
    # 检查图像尺寸 - 正确解析BCHW格式
    batch_size, channels, height, width = img.shape
    print(f"图像尺寸: {batch_size}x{channels}x{height}x{width}")
    
    # 记录每个通道的原始最大最小值（用于反归一化）
    channel_stats = []
    for i in range(channels):
        channel_data = img[0, i, :, :]  # 提取单个通道
        min_val = channel_data.min().item()
        max_val = channel_data.max().item()
        channel_stats.append({'min': min_val, 'max': max_val})
        print(f"通道 {i+1} 原始范围: [{min_val:.3f}, {max_val:.3f}]")
    
    # 对每个通道进行独立的最大最小值归一化
    img_normalized = img.clone()
    for i in range(channels):
        min_val = channel_stats[i]['min']
        max_val = channel_stats[i]['max']
        if max_val > min_val:  # 避免除零
            img_normalized[0, i, :, :] = (img[0, i, :, :] - min_val) / (max_val - min_val)
        print(f"通道 {i+1} 归一化后范围: [0.000, 1.000]")
    
    # 如果图像太小，进行填充
    if height < 4 or width < 4:
        print(f"图像尺寸太小 ({height}x{width})，进行填充...")
        pad_h = max(0, 8 - height)
        pad_w = max(0, 8 - width)
        img_normalized = F.pad(img_normalized, (0, pad_w, 0, pad_h), 'reflect')
        print(f"填充后尺寸: {img_normalized.shape[2]}x{img_normalized.shape[3]}")
    
    # 确保尺寸能被4整除
    _, _, h, w = img_normalized.shape
    if h % 4 != 0 or w % 4 != 0:
        print(f"图像尺寸不能被4整除 ({h}x{w})，进行调整...")
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        img_normalized = F.pad(img_normalized, (0, pad_w, 0, pad_h), 'reflect')
        print(f"调整后尺寸: {img_normalized.shape[2]}x{img_normalized.shape[3]}")
    
    # 设置模型输入
    model.lq = img_normalized
    model.gt = img_normalized
    
    # 模型推理
    model.test()
    
    # 获取输出
    output = model.output
    
    # 如果之前进行了填充，现在需要裁剪回原始尺寸
    if height < 4 or width < 4:
        output = output[:, :, :height, :width]
    
    # 对每个通道进行反归一化
    output_denormalized = output.clone()
    for i in range(channels):
        min_val = channel_stats[i]['min']
        max_val = channel_stats[i]['max']
        if max_val > min_val:
            output_denormalized[0, i, :, :] = output[0, i, :, :] * (max_val - min_val) + min_val
        print(f"通道 {i+1} 反归一化后范围: [{output_denormalized[0, i, :, :].min().item():.3f}, {output_denormalized[0, i, :, :].max().item():.3f}]")
    
    # 保存结果为mat文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将输出转换为numpy数组，保持CHW格式
    output_np = output_denormalized.detach().cpu().numpy()  # BCHW -> numpy
    output_np = output_np[0]  # 移除batch维度，保持CHW格式
    
    # 保存为mat文件，保持CHW格式
    try:
        import scipy.io as sio
        sio.savemat(output_path, {'img': output_np})
        print(f"处理结果已保存到: {output_path} (CHW格式)")
        print(f"输出数据形状: {output_np.shape} (通道数x高度x宽度)")
    except ImportError:
        print("警告: 需要安装scipy来保存.mat文件")
        # 如果无法保存mat文件，保存为numpy文件
        np.save(output_path.replace('.mat', '.npy'), output_np)
        print(f"处理结果已保存为numpy文件: {output_path.replace('.mat', '.npy')} (CHW格式)")
    
    return output_np

def process_batch_images(model, input_dir, output_dir, device='cuda'):
    """批量处理图像"""
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图像格式
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.mat']
    
    processed_count = 0
    failed_count = 0
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        # 检查文件格式
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            # 生成输出文件名 - 保持原始格式
            name, ext = os.path.splitext(filename)
            if ext.lower() == '.mat':
                output_filename = f"{name}_processed.mat"  # 保持mat格式
            else:
                output_filename = f"{name}_processed.mat"  # 其他格式也保存为mat
                
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                process_single_image(model, file_path, output_path, device)
                processed_count += 1
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
                failed_count += 1
                # 尝试获取更详细的错误信息
                import traceback
                print(f"详细错误信息:")
                traceback.print_exc()
    
    print(f"批量处理完成，成功处理 {processed_count} 张图像，失败 {failed_count} 张图像")
    print(f"每个输入图像会生成1个输出文件：")
    print(f"  - 1个3通道mat文件 (xxx_processed.mat) - CHW格式保存")

def main():
    parser = argparse.ArgumentParser(description='S2模型单独处理噪声图像')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='S2模型路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像路径或目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出路径或目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备类型 (cuda/cpu)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"模型文件不存在: {args.model_path}")
        return
    
    # 创建配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            opt = yaml.safe_load(f)
    else:
        opt = create_s2_only_config()
        # 更新模型路径
        opt['path']['pretrain_network_g'] = args.model_path
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = S2OnlyModel(opt)
    # 移除这行，因为SRModel基类已经处理了设备移动
    # model.to(device)
    
    # 加载预训练模型
    print(f"正在加载S2模型: {args.model_path}")
    model.load_network(model.net_g, args.model_path, True, 'params_ema')
    print("S2模型加载成功!")
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 单文件处理
        process_single_image(model, args.input, args.output, device)
    elif os.path.isdir(args.input):
        # 批量处理
        process_batch_images(model, args.input, args.output, device)
    else:
        print(f"输入路径不存在: {args.input}")

if __name__ == '__main__':
    main() 
