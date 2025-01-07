#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

# resnet-18, fpn=[128,256,512]
# WORK_DIR=work_dirs/debug
# CONFIG=plugin/LaneSegNet/configs/lanesegnet_r18_3x1_24e_olv2_subset_A_debug.py

# # resnet-50,
# WORK_DIR=work_dirs/lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket_debug
# CONFIG=plugin/LaneSegNet/configs/lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket_debug.py


# WORK_DIR=work_dirs/debug
# CONFIG=plugin/LaneSegNet/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py

WORK_DIR=work_dirs/lanesegnet_vmamba_2x2_debug
CONFIG=plugin/LaneSegNet_VMamba/configs/lanesegnet_vmamba_2x2_24e_olv2_subset_A.py

GPUS=$1
PORT=${PORT:-28508}
CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train_ls.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
