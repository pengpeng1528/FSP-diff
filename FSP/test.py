# flake8: noqa
import os.path as osp
import sys
from basicsr.test import test_pipeline

project_root = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.insert(0, project_root)

import FSP.archs
import FSP.data
import FSP.models

if __name__ == '__main__':
    root_path = project_root
    test_pipeline(root_path)
