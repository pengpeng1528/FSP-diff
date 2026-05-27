import os
import numpy as np
from scipy.io import loadmat, savemat
import odl

# 配置参数
input_folder = r"/SSD_8T/pengpeng/DiffIR-master03/allfen"  # 输入mat文件存放的文件夹
output_folder = r"/SSD_8T/pengpeng/DiffIR-master03/allfenpro"  # 输出结果保存路径
os.makedirs(output_folder, exist_ok=True)  # 自动创建输出目录

# 创建扇束CT算子
geometry = odl.tomo.FanBeamGeometry(
    odl.uniform_partition(0, 2 * np.pi, 512),  # 角度范围
    odl.uniform_partition(-360, 360, 512),  # 探测器范围
    src_radius=500, det_radius=500  # 几何参数
)
reco_space = odl.uniform_discr([-128, -128], [128, 128], (512, 512))  # 重建空间
ray_transform = odl.tomo.RayTransform(reco_space, geometry)  # 投影算子


def process_mat_file(mat_path):
    """处理单个mat文件的核心函数"""
    # 加载数据并验证维度
    data = loadmat(mat_path)['rec_img']  # 假设数据变量名为"sinogram"
    print(f"Loaded data shape: {data.shape}")  # 添加打印语句，检查加载的数据形状

    assert data.shape == (512, 512), f"Invalid data shape in {mat_path}"

    # 对整个512x512的图像切片生成投影（不进行任何预处理或归一化）
    proj_data = ray_transform(data).data
    proj_data = proj_data.astype(np.float32)

    # 生成输出文件名（保持原文件名不变）
    base_name = os.path.basename(mat_path)
    output_path = os.path.join(output_folder, base_name)

    # 保存为 .mat 文件
    savemat(output_path, {'img': proj_data})  # 以原始变量名保存


def main():
    # 遍历输入文件夹中的所有mat文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".mat"):
            mat_path = os.path.join(input_folder, filename)
            print(f"Processing: {mat_path}")
            process_mat_file(mat_path)

    print("所有文件处理完成")


if __name__ == "__main__":
    main()