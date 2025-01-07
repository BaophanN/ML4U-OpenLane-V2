!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

# WORK_DIR=work_dirs/toponet_separated
# CONFIG=projects/configs/toponet_separated.py
WORK_DIR=work_dirs/toponet_r18_1_24e_olv2_subset_A
CONFIG=plugin/TopoNet/configs/toponet_r18_1_24e_olv2_subset_A.py


GPUS=$1
PORT=${PORT:-28510}

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log


