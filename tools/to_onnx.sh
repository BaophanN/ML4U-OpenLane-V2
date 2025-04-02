#!/usr/bin/env bash

set -x
config=plugin/LaneSegNet/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py
checkpoint=lanesegnet_r50_8x1_24e_olv2_subset_A.pth
work_dir=work_dirs/LaneSegNet 

# config=configs/fastbev/paper/fastbev-r50-cbgs.py
# checkpoint=work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth
# work_dir=work_dirs/

DEUBUG=1 python tools/convert_lanesegnet_to_onnx.py $config $checkpoint $work_dir --dynamic