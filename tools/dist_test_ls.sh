#!/usr/bin/env bash
clear
set -x

timestamp=$(date +"%y%m%d.%H%M%S")

# WORK_DIR=/workspace/log/work_dirs/pretrained
# CONFIG=plugin/LaneSegNet/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py
# CHECKPOINT=lanesegnet_r50_8x1_24e_olv2_subset_A.pth

WORK_DIR=work_dirs/LaneSegNet

CONFIG=plugin/LaneSegNet_OG/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py
CHECKPOINT=lanesegnet_r50_8x1_24e_olv2_subset_A.pth
# CHECKPOINT=${WORK_DIR}/epoch_14.pth

GPUS=$1
PORT=${PORT:-28522}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test_ls.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 ${@:2} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log