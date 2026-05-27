import os
import odl
import numpy as np
from scipy.io import loadmat, savemat

# -----------------------------------------------------------------------------
# 本脚本批量读取一个文件夹下的投影 (sinogram) .mat 文件，
# 使用 FBP 算法重建为单精度图像，并保存到指定输出目录。
# 仅执行重建操作，投影已在 .mat 文件中给出。
# -----------------------------------------------------------------------------

# 输入、输出目录（按需修改）
input_dir = r"/SSD_8T/pengpeng/DiffIR-master03/allquan3000"   # 存放投影 .mat 文件
output_dir = r"/SSD_8T/pengpeng/DiffIR-master03/allquan3000rec"              # 保存重建结果

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# --------------------------- 几何及算子初始化 ---------------------------
# 根据具体扫描参数设置 FanBeamGeometry，这里保持与原脚本一致
Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, 512)
Fan_detector_partition = odl.uniform_partition(-360, 360, 512)
Fan_geometry = odl.tomo.FanBeamGeometry(
    Fan_angle_partition, Fan_detector_partition, src_radius=500, det_radius=500
)

Fan_reco_space = odl.uniform_discr(
    min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype="float32"
)

Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry)
Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)  # FBP 重建算子

# --------------------------- 批量处理开始 ---------------------------

mat_files = [f for f in os.listdir(input_dir) if f.endswith(".mat")]
if not mat_files:
    raise FileNotFoundError(f"在输入目录 {input_dir} 中未找到 .mat 文件")

for fname in mat_files:
    in_path = os.path.join(input_dir, fname)
    # 读入投影数据：假设变量名为 'img'，如果不同请修改下行
    mat_data = loadmat(in_path)
    if "img" not in mat_data:
        raise KeyError(f"文件 {fname} 中未找到键 'img'，请检查变量名")

    sinogram = mat_data["img"].astype(np.float32)
    sinogram = np.squeeze(sinogram)

    # 处理单通道或多通道
    if sinogram.ndim == 2:
        # FBP 重建 -> ndarray(float32)
        rec_img = Fan_FBP(sinogram).data.astype(np.float32)
    elif sinogram.ndim == 3:
        # 自动判断通道维度，并逐通道重建
        nA = Fan_angle_partition.size
        nD = Fan_detector_partition.size
        shape = sinogram.shape
        # 选择既不是角度数也不是探测器数的维度作为通道维
        candidate_axes = [ax for ax, s in enumerate(shape) if s != nA and s != nD]
        if len(candidate_axes) >= 1:
            ch_axis = candidate_axes[0]
        else:
            # 回退策略：若三维都等于(nA或nD)，默认将轴0作为通道维
            ch_axis = 0
        num_channels = shape[ch_axis]

        # 为输出分配数组，保持与输入相同的通道布局
        out_shape = list(shape)
        out_arr = np.zeros(out_shape, dtype=np.float32)

        for ci in range(num_channels):
            sino_ci = np.take(sinogram, ci, axis=ch_axis)
            if sino_ci.ndim != 2:
                raise ValueError(f"提取到的单通道正弦图维度不是2D: shape={sino_ci.shape}")
            rec_ci = Fan_FBP(sino_ci).data.astype(np.float32)
            # 放回对应通道位置
            out_arr = np.swapaxes(out_arr, 0, ch_axis)
            out_arr[ci, ...] = rec_ci
            out_arr = np.swapaxes(out_arr, 0, ch_axis)
        rec_img = out_arr
    else:
        raise ValueError(f"不支持的正弦图维度: {sinogram.ndim}, shape={sinogram.shape}")

    # 保存，文件名保持一致
    base_name = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, f"{base_name}.mat")
    savemat(out_path, {"rec_img": rec_img})

    print(f"已重建并保存: {out_path}  | 输入shape={sinogram.shape} 输出shape={rec_img.shape}")

# -----------------------------------------------------------------------------
# 运行示例（Linux 终端）
#   python pro1.py
# -----------------------------------------------------------------------------