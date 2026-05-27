import importlib
from basicsr.utils import scandir
from os import path as osp

# automatically scan and import loss modules for registry
# scan all the files that end with '_loss.py' under the losses folder
loss_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
# import all the loss modules
_arch_modules = [importlib.import_module(f'.{file_name}', package=__name__) for file_name in arch_filenames]
