#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

<<<<<<< HEAD
###############################################################
# Set the plugin folder and config name only
# 
PLUGIN=LaneSegNet_FastBEV
CONFIG_NAME=lanesegnet_r18_1x1_1e_olv2_subset_A
################################################################
WORK_DIR=work_dirs/${PLUGIN}/${CONFIG_NAME}_debug
# WORK_DIR=work_dirs/debug
CONFIG=plugin/${PLUGIN}/configs/${CONFIG_NAME}.py
=======
# WORK_DIR=work_dirs/lanesegnet_streaming
# CONFIG=plugin/LaneSegNet_Streaming/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py

WORK_DIR=work_dirs/lanesegnet_r18_lss_1x1_1e_olv2_subset_A
CONFIG=plugin/LaneSegNet_BEVFusion/configs/lanesegnet_r18_lss_1x1_1e_olv2_subset_A.py
>>>>>>> 06e1c1a6379e75fc045ae2e2b10a6e8029baa6f4

GPUS=$1
PORT=${PORT:-28511}
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train_ls.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
