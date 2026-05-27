#!/bin/bash

# S2模型单独处理图像的运行脚本

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 模型路径
MODEL_PATH="/SSD_8T/pengpeng/DiffIR-master03/FSP-diff/modelbaocun/500rec/s2/net_g_130000.pth"

# 输入和输出路径
INPUT_PATH="/SSD_8T/彭鹏/数据/allfentu500/val"
OUTPUT_PATH="/SSD_8T/彭鹏/数据/allfentu500/valdenoisy"

# 创建输出目录
mkdir -p $OUTPUT_PATH

# 运行S2模型处理
python test_s2_only.py \
    --model_path $MODEL_PATH \
    --input $INPUT_PATH \
    --output $OUTPUT_PATH \
    --device cuda

echo "处理完成！结果保存在: $OUTPUT_PATH" 