#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`
###############################################################
# Set the plugin folder and config name only
# 
PLUGIN=LaneSegNet
CONFIG_NAME=lanesegnet_r50_8x1_24e_olv2_subset_A
################################################################
WORK_DIR=work_dirs/${PLUGIN}/${CONFIG_NAME}_debug
# WORK_DIR=work_dirs/debug
CONFIG=plugin/${PLUGIN}/configs/${CONFIG_NAME}.py

GPUS=$1
PORT=${PORT:-28511}
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train_ls.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
