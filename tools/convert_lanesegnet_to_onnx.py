import sys
sys.path.insert(0,'/workspace/source/Mapless')
import argparse
import torch.onnx
from onnxsim import simplify
from mmcv import Config
from mmcv.parallel import DataContainer
try:

    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import os
from typing import Optional,Union

import numpy as np
import onnx

import torch
from mmcv.runner import load_checkpoint

from packaging import version
from torch.utils.data import DataLoader

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='Export LaneSegNet to ONNX')
    parser.add_argument('config', help='Path to model config file')
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('work_dir', help='Directory to save ONNX model')
    parser.add_argument('--prefix', default='lanesegnet', help='ONNX filename prefix')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic batch size')
    args = parser.parse_args()
    return args

def prepare_img_metas(img_metas): 
    """
    img_metas: [{batch1}, {batch_2}]

    can_bus: numpy.ndarray[ 7.14223950e+02  3.03365031e+03 -2.34128162e+01  7.18497279e-01
    7.18497279e-01  7.18497279e-01  7.18497279e-01  0.00000000e+00
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    4.74515919e+00  2.71877595e+02]: numpy.ndarray 

    [[ 3.27478053e-02  9.98958447e-01 -3.17742317e-02]
    [-9.99463461e-01  3.27118481e-02 -1.65095503e-03]
    [-6.09841631e-04  3.18112488e-02  9.99493708e-01]]
    lidar2global_rotation: 



    """
    can_bus = torch.tensor([meta['can_bus'] for meta in img_metas],  # list of [x, y, z]
    dtype=torch.float32)

    lidar2global_rotation = torch.tensor([meta['lidar2global_rotation'] for meta in img_metas])

    return can_bus, lidar2global_rotation
def main():
    args = parse_args()
    
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # Load configuration
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'  # Mark for ONNX export
    print(cfg.model.type)
    cfg.gpu_ids = [0]
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    # Modify data pipeline for testing
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    dataset = build_dataset(cfg.data.test)
    
    dataloader = build_dataloader(dataset, **test_loader_cfg)

    # Load LaneSegNet model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.forward = model.forward_trt
    model.cuda().eval()

    for i, data in enumerate(dataloader):

        img_metas = data['img_metas'].data[0]
        img = data['img'].data[0] # Extract first image batch
        # print(type(img))
        # img_metas = model.prepare_img_metas(img_metas)
        can_bus, lidar2global_rotation = prepare_img_metas(img_metas)

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        
        """
        1 sample data: 7 images + meta data 
        """
        # print(type(img.float()));exit()
        img = img.cuda()
        can_bus = can_bus.float().cuda()
        lidar2global_rotation = lidar2global_rotation.float().cuda()
        with torch.no_grad():

            output_names = ['all_cls_scores'
                        'all_lanes_preds',
                        'all_mask_preds',
                        'all_lanes_left_type',
                        'all_lanes_right_type',
                        'history_states']

            # Dynamic shape support
            dynamic_axes = {'img': {0: 'batch_size'}}
            if args.dynamic:
                print("✅ Enabling dynamic batch size for ONNX export")


            onnx_path = os.path.join(args.work_dir, f"{args.prefix}.onnx")
            torch.onnx.export(
                model,
                (img, can_bus, lidar2global_rotation), 
                'lanesegnet.onnx',
                onnx_path,
                opset_version=11,
                input_names=['img', 'can_bus','lidar2global_rotation'],
                output_names=output_names,
                dynamic_axes=dynamic_axes if args.dynamic else None
            )
            print(f"🚀 ONNX model saved to {onnx_path}")
        img.to('cpu')
        lidar2global_rotation.to('cpu')
        can_bus.to('cpu')
        break  # Process only the first batch

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX Model is valid")
    except Exception as e:
        print("❌ ONNX Model check failed:", str(e))

    # Simplify ONNX
    onnx_simp, check = simplify(onnx_model)
    assert check, "❌ ONNX simplification failed"
    simplified_onnx_path = os.path.join(args.work_dir, f"{args.prefix}_simplified.onnx")
    onnx.save(onnx_simp, simplified_onnx_path)
    print(f"✅ Simplified ONNX model saved: {simplified_onnx_path}")


if __name__ == '__main__':
    main()
