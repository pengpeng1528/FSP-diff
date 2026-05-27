import numpy as np
import scipy.io as sio
import os
import re

# 硬编码输入输出路径
input_folder_path = r'/SSD_8T/pengpeng/DiffIR-master03/allfenpro'  # 存放分层.mat文件的文件夹
output_folder_path = r'/SSD_8T/pengpeng/DiffIR-master03/allquan'  # 保存合并结果的文件夹

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 获取所有符合n_n命名规则的.mat文件（排除隐藏文件）
file_list = [f for f in os.listdir(input_folder_path) 
             if re.match(r'^\d+_\d+\.mat$', f) and not f.startswith('.')]

# 提取唯一基础名（如200_1.mat → 200）
base_names = set()
for file in file_list:
    match = re.match(r'^(\d+)_\d+\.mat$', file)
    if match:
        base_names.add(match.group(1))
base_names = sorted(base_names)  # 转为有序列表

def process_base(base_name):
    """处理单个基础名对应的文件组"""
    print(f'\n正在处理: {base_name}')
    
    # 查找当前基础名的6个通道文件（n_1到n_6）
    channel_files = []
    valid_channels = [True] * 6
    for chan in range(1, 7):  # 通道1到6
        file_name = f"{base_name}_{chan}.mat"
        file_path = os.path.join(input_folder_path, file_name)
        
        if os.path.exists(file_path):
            channel_files.append(file_path)
        else:
            print(f'缺失通道文件: {file_name}')
            valid_channels[chan-1] = False
    
    # 检查是否所有通道文件都存在
    if not all(valid_channels):
        print(f'跳过 {base_name}：缺少{sum(not v for v in valid_channels)}个通道文件')
        return
    
    # 初始化合并数据容器
    merged_data = np.zeros((512, 512, 6), dtype=np.float64)
    
    # 加载并验证所有通道数据
    for idx, file_path in enumerate(channel_files):
        try:
            data = sio.loadmat(file_path)
            if 'img' not in data:
                raise ValueError('变量img不存在')
            
            img_layer = data['img'].astype(np.float64)
            if img_layer.shape != (512, 512):
                raise ValueError(f'尺寸应为512×512，实际为{img_layer.shape}')
            merged_data[:, :, idx] = img_layer / 10  # 先除10预处理
        except Exception as e:
            print(f'文件 {file_path} 加载失败: {str(e)}')
            valid_channels[idx] = False
    
    # 如果存在加载失败的通道则跳过
    if not all(valid_channels):
        print(f'跳过 {base_name}：{sum(not v for v in valid_channels)}个通道数据无效')
        return
    
    # 光子数计算（修改后逻辑）
    scale_factor = 10000  # 新缩放因子
    g_dChanPhon = np.zeros((512, 512, 6), dtype=np.float64)
    for nIndex in range(6):
        g_dChanPhon[:, :, nIndex] = np.exp(-merged_data[:, :, nIndex]) * scale_factor
    
    g_dAllPhon = np.sum(g_dChanPhon, axis=2)
    g_dAll0Phon = scale_factor * 6
    g_dAllProj = -np.log(g_dAllPhon / g_dAll0Phon)  # 移除max限制
    g_dAllProj[g_dAllPhon <= 0] = 0  # 仅处理非正值
    
    # 后处理：乘以10恢复量级并转换为float32类型
    img = (g_dAllProj * 10).astype(np.float32)  # 输出变量名改为img
    
    # 保存合并结果（使用v7格式支持大文件）
    output_path = os.path.join(output_folder_path, f"{base_name}.mat")
    sio.savemat(output_path, {'img': img}, do_compression=True)  # 等效于MATLAB的-v7
    print(f'成功保存: {base_name}.mat')

# 处理每个基础名对应的文件组
for base in base_names:
    process_base(base)

print('===== 所有文件处理完成 =====')