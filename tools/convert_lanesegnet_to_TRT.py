import argparse

import torch.onnx
from onnxsim import simplify

from mmcv import Config
# from mmdeploy.backend.tensorrt.utils import save, search_cuda_version

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import os
from typing import Dict, Optional, Sequence, Union

import h5py
import mmcv
import numpy as np
import onnx
# import pycuda.driver as cuda
# import tensorrt as trt
import torch
import tqdm
from mmcv.runner import load_checkpoint
# from mmdeploy.apis.core import no_mp
# from mmdeploy.backend.tensorrt.calib_utils import HDF5Calibrator
# from mmdeploy.backend.tensorrt.init_plugins import load_tensorrt_plugin
# from mmdeploy.utils import load_config
from packaging import version
from torch.utils.data import DataLoader

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module