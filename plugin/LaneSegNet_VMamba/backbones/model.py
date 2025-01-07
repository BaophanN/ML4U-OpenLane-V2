import os
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.utils import checkpoint

# from mmengine.model import BaseModule
from mmcv.runner.base_module import BaseModule
f
# from mmdet.registry import MODELS as MODELS_MMDET
# from mmseg.registry import MODELS as MODELS_MMSEG
from mmdet.models import BACKBONES

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

# classification/models 
build = import_abspy(
    "backbones", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../LaneSegNet_VMamba/"),
)
Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM


# this can be inside backbones 
# @MODELS_MMSEG.register_module()
# @MODELS_MMDET.register_module()
@BACKBONES.register_module()
class MM_VSSM(BaseModule, Backbone_VSSM):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_VSSM.__init__(self, *args, **kwargs)

current_directory = os.path.dirname(os.path.abspath(__file__))


