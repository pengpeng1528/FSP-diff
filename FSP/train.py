import os
# flake8: noqa
import os.path as osp
import sys

# 只暴露项目根目录，统一通过 `FSP.*` 导入，避免重复注册。
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = osp.abspath(osp.join(current_dir, osp.pardir))
sys.path.insert(0, project_root)

from FSP.train_pipeline import train_pipeline

import FSP.archs
import FSP.data
import FSP.models
import FSP.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
