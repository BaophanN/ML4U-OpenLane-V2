#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

# WORK_DIR=work_dirs/baseline_large_separated
# CONFIG=plugin/SeparatedNet/configs/baseline_large_separated.py
# CHECKPOINT=${WORK_DIR}/latest.pth

WORK_DIR=work_dirs/toponet_r18_1_24e_olv2_subset_A
CONFIG=plugin/TopoNet/configs/toponet_r18_1_24e_olv2_subset_A.py
CHECKPOINT=${WORK_DIR}/latest.pth

# WORK_DIR=work_dirs/toponet_resnet50v1
# CONFIG=plugin/TopoNet/configs/toponet_r50_1_24e_olv2_subset_A.pth
# CHECKPOINT=${WORK_DIR}/epoch_24_resnet50v1.pth

GPUS=$1
PORT=${PORT:-28510}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/testv2.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 ${@:2} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log