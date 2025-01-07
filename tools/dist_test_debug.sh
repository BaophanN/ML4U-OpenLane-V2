#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

# WORK_DIR=/workspace/source/work_dirs/lanesegnet_r18_2x2_24e_fpn128_256_512
# CONFIG=plugin/LaneSegNet/configs/lanesegnet_r18_2x2_24e_fpn128_256_512.py

WORK_DIR=work_dirs/lanesegnet_r50_8x1_24e_olv2_video_test_mode
CONFIG=plugin/LaneSegNet_LSS_Topologic/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py
CHECKPOINT=/workspace/source/work_dirs/lanesegnet_r50_8x1_24e_olv2_subset_A/epoch_14.pth




GPUS=$1
PORT=${PORT:-28520}
CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test_ls.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 ${@:2} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log
