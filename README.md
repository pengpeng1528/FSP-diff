# FSP-diff 使用说明

这个仓库当前采用两阶段训练流程：

- `trainS1.sh`：训练 Stage 1 (`FSPS1`)
- `trainS2.sh`：训练 Stage 2 (`FSPS2`)
- `run_s2_only.sh`：使用训练好的 S2 模型单独做推理

下面的说明基于仓库当前脚本和配置文件整理，可直接对照使用。

## 1. 主要文件

- `trainS1.sh`：S1 训练入口
- `trainS2.sh`：S2 训练入口
- `run_s2_only.sh`：S2 单独测试/推理入口
- `bin/FullSpectrum_Merge.py`：能级融合代码
- `bin/compare_metricsall.py`：指标计算代码
- `options/train_FSPS1.yml`：S1 训练配置
- `options/train_FSPS2.yml`：S2 训练配置
- `test_s2_only.py`：S2 单独推理脚本
- `test/data/val`：测试用噪声验证集
- `test/data/cleanval/val`：测试用干净标签
- `test/out/valdenoisy`：当前测试输出目录

## 2. 环境准备

建议先准备好与本机 CUDA 对应的 PyTorch 环境，然后在仓库根目录安装依赖。

```bash
pip install -r requirements.txt
python setup.py develop
```

说明：

- 仓库完整环境以 `requirements.txt` 为准
- `test_s2_only.py` 读写 `.mat` 文件需要 `scipy`，该依赖已包含在 `requirements.txt` 中
- `python setup.py develop` 用于把当前项目按开发模式安装

### 2.1 当前环境

当前仓库的 `requirements.txt` 已固定了一套环境，核心依赖包括：

- `torch==2.1.2`
- `torchvision==0.16.2`
- `basicsr==1.4.2`
- `numpy==1.24.3`
- `scipy==1.10.1`
- `scikit-image==0.21.0`
- `opencv-python==4.9.0.80`
- `pillow==10.3.0`
- `PyYAML==6.0.1`
- `einops==0.7.0`
- `pandas==2.0.3`
- `tqdm==4.66.2`
- `albumentations==0.5.2`
- `detectron2==0.6+cu113`
- `astra-toolbox==2.2.0`

此外，`requirements.txt` 中还包含：

- `tensorflow==2.13.1`
- `pytorch-lightning==1.2.9`
- `tensorboard==2.13.0`
- `wandb==0.21.4`

如果你希望完全复现当前环境，直接使用：

```bash
pip install -r requirements.txt
```

如果你是在另一台机器上重新配环境，建议优先检查以下兼容性：

- `torch` / `torchvision`
- CUDA 相关 `nvidia-*` 包
- `detectron2`

## 3. 数据组织

当前配置文件里使用的是成对数据：

- `GT / clean`：清晰图像
- `LQ / noisy`：退化或含噪图像

仓库当前默认路径如下：

```text
GT:
/SSD_8T/彭鹏/数据/allfentu/train
/SSD_8T/彭鹏/数据/allfentu/val

LQ:
/SSD_8T/彭鹏/数据/allfentu2000/train
/SSD_8T/彭鹏/数据/allfentu2000/val
```

建议保持下面这种目录结构：

```text
/path/to/allfentu/
  train/
  val/

/path/to/allfentu2000/
  train/
  val/
```

要求：

- `GT` 和 `LQ` 文件名一一对应
- 如果你的数据路径不同，需要先修改 `options/train_FSPS1.yml` 和 `options/train_FSPS2.yml` 里的 `dataroot_gt`、`dataroot_lq`

## 4. 训练流程

### 4.1 训练 Stage 1

`trainS1.sh` 的实际命令是：

```bash
CUDA_VISIBLE_DEVICES=0 python3 FSP/train.py -opt options/train_FSPS1.yml --launcher none
```

运行方式：

```bash
sh trainS1.sh
```

训练前请确认：

1. `options/train_FSPS1.yml` 中的训练集和验证集路径正确
2. 如果要换 GPU，修改 `trainS1.sh` 里的 `CUDA_VISIBLE_DEVICES`

补充说明：

- 当前脚本按单卡方式运行
- `train_FSPS1.yml` 里的实验名是 `train_FSPS1`
- 训练结果通常会保存在 `experiments/train_FSPS1/` 下
- 配置中 `save_checkpoint_freq` 为 `400`，因此会按设定频率保存模型

### 4.2 训练 Stage 2

`trainS2.sh` 的实际命令是：

```bash
CUDA_VISIBLE_DEVICES=0 python3 FSP/train.py -opt options/train_FSPS2.yml --launcher none
```

运行方式：

```bash
sh trainS2.sh
```

训练前请确认：

1. `options/train_FSPS2.yml` 中的训练集和验证集路径正确
2. `path.pretrain_network_g` 指向可用的 S1 权重
3. `path.pretrain_network_S1` 指向同一个或对应的 S1 权重
4. 如果要换 GPU，修改 `trainS2.sh` 里的 `CUDA_VISIBLE_DEVICES`

当前配置里这两个字段都需要手动检查：

```yaml
path:
  pretrain_network_g: /path/to/your/S1_checkpoint.pth
  pretrain_network_S1: /path/to/your/S1_checkpoint.pth
```

如果你先在本仓库完成了 S1 训练，通常可以把它们改成类似下面的路径：

```text
experiments/train_FSPS1/models/net_g_xxx.pth
```

补充说明：

- 当前脚本同样按单卡方式运行
- `train_FSPS2.yml` 里的实验名是 `train_FSPS2`
- 训练结果通常会保存在 `experiments/train_FSPS2/` 下

## 5. S2 单独推理 / 测试

`run_s2_only.sh` 用于不依赖 S1、直接调用训练好的 S2 模型做推理。

当前脚本内容对应的关键参数为：

```bash
MODEL_PATH="/SSD_8T/pengpeng/DiffIR-master03/FSP-diff/modelbaocun/500rec/s2/net_g_130000.pth"
INPUT_PATH="/SSD_8T/彭鹏/数据/allfentu500/val"
OUTPUT_PATH="/SSD_8T/彭鹏/数据/allfentu500/valdenoisy"
```

运行方式：

```bash
sh run_s2_only.sh
```

脚本实际调用的是：

```bash
python test_s2_only.py \
    --model_path /path/to/net_g_xxx.pth \
    --input /path/to/input \
    --output /path/to/output \
    --device cuda
```

如果你的模型或数据路径不同，先修改 `run_s2_only.sh` 中的：

- `MODEL_PATH`
- `INPUT_PATH`
- `OUTPUT_PATH`

### 5.1 推理输入

`test_s2_only.py` 支持以下输入格式：

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tiff`
- `.mat`

可以输入：

- 单张图像文件
- 一个目录，脚本会批量处理目录下所有支持的文件

### 5.2 推理输出

输出结果默认保存为 `.mat` 文件，命名方式为：

```text
原文件名_processed.mat
```

例如：

```text
84_1.mat -> 84_1_processed.mat
image.png -> image_processed.mat
```

输出数据说明：

- 保存变量名为 `img`
- 数据格式为 `CHW`
- 即 `通道数 x 高度 x 宽度`

## 6. 直接命令示例

### 6.1 训练 S1

```bash
sh trainS1.sh
```

### 6.2 训练 S2

```bash
sh trainS2.sh
```

### 6.3 批量推理

```bash
sh run_s2_only.sh
```

### 6.4 不通过 shell 脚本直接推理

```bash
python test_s2_only.py \
  --model_path ./modelbaocun/500rec/s2/net_g_130000.pth \
  --input /path/to/input_dir \
  --output /path/to/output_dir \
  --device cuda
```

## 7. test 目录说明

当前仓库里已经放好了一个可直接对照的测试目录：

```text
test/
  data/
    val/                 # 噪声 val
    cleanval/val/        # 干净标签
  out/
    valdenoisy/          # 推理输出
```

其中：

- `test/data/val` 中是噪声输入 `.mat`
- `test/data/cleanval/val` 中是对应的干净标签 `.mat`
- `test/out/valdenoisy` 中是模型输出结果，文件名形如 `xxx_processed.mat`

如果你想直接按仓库现有测试样例走一遍，可以把 `run_s2_only.sh` 的输入输出路径改到这个目录体系。

## 8. 能级融合

`bin/FullSpectrum_Merge.py` 是能级融合脚本，用于把同一个样本的多能级结果合并成一个最终投影结果。

脚本当前默认使用硬编码路径：

```python
input_folder_path = '/SSD_8T/pengpeng/DiffIR-master03/allfenpro'
output_folder_path = '/SSD_8T/pengpeng/DiffIR-master03/allquan'
```

运行前需要先按你的数据修改这两个路径，然后执行：

```bash
python bin/FullSpectrum_Merge.py
```

输入要求：

- 输入为 `.mat` 文件
- 文件名格式必须是 `n_1.mat` 到 `n_6.mat`
- 例如同一个样本需要有 `200_1.mat`、`200_2.mat`、...、`200_6.mat`
- 每个文件中需要包含变量 `img`
- 当前脚本默认每个 `img` 的大小为 `512x512`

输出结果：

- 每 6 个通道文件融合成 1 个文件
- 输出命名为 `n.mat`
- 结果保存在 `output_folder_path`

## 9. 指标计算

`bin/compare_metricsall.py` 用于计算输出结果和干净标签之间的 `PSNR` 与 `SSIM`。

脚本当前默认比较的目录是：

```text
clean_dir = /SSD_8T/pengpeng/DiffIR-master03/FSP-diff/test/data/cleanval/val
noisy_dir = /SSD_8T/pengpeng/DiffIR-master03/FSP-diff/test/out/valdenoisy
```

运行方式：

```bash
python bin/compare_metricsall.py
```

匹配规则：

- 脚本会在两个目录中查找同名 `.mat` 文件
- 输出文件如果带 `_processed` 后缀，会自动去掉后再和 GT 对齐
- 例如 `84_1.mat` 会匹配 `84_1_processed.mat`

指标输出：

- 控制台会打印每个文件、每个通道的 `PSNR` 和 `SSIM`
- 会统计每个通道的平均值
- 会统计全局平均 `PSNR` 和 `SSIM`
- 日志会保存到 `results/testlog/metrics.txt`

说明：

- 脚本支持单通道和多通道 `.mat`
- 多通道情况下按通道分别计算指标
- 计算前会对 GT 和结果做联合归一化

## 10. 注意事项

- 这个仓库里的很多路径是绝对路径，换机器后需要先改配置
- `trainS2.sh` 训练前，必须确认 S1 预训练权重路径有效
- 默认脚本使用 `CUDA_VISIBLE_DEVICES=0`，如果要换卡需要手动修改
- `run_s2_only.sh` 是 S2 单独推理，不走 S1 联合测试流程
- 如果使用 `.mat` 数据，请确认环境里已安装 `scipy`
- `FullSpectrum_Merge.py` 和 `compare_metricsall.py` 都写了绝对路径，运行前建议先检查
- `FullSpectrum_Merge.py` 默认要求 6 个能级文件齐全，且每张图大小是 `512x512`

## 11. 推荐使用顺序

1. 修改 `options/train_FSPS1.yml` 中的数据路径
2. 运行 `sh trainS1.sh`
3. 记录 S1 输出的 checkpoint 路径
4. 修改 `options/train_FSPS2.yml` 中的 `pretrain_network_g` 和 `pretrain_network_S1`
5. 运行 `sh trainS2.sh`
6. 修改 `run_s2_only.sh` 中的模型、输入、输出路径
7. 运行 `sh run_s2_only.sh` 完成推理
8. 如果需要多能级合并，修改并运行 `python bin/FullSpectrum_Merge.py`
9. 如果需要评估结果，运行 `python bin/compare_metricsall.py`
