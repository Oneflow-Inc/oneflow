#!/bin/bash

set -xeu

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
ONEFLOW_CI_DEVICE_NUMS=${ONEFLOW_CI_DEVICE_NUMS:-"1 2 4"}

for device_num in ${ONEFLOW_CI_DEVICE_NUMS}
do
    export ONEFLOW_TEST_NODE_NUM=2
    export ONEFLOW_TEST_DEVICE_NUM=$device_num
    time python3 ${src_dir}/ci/test/multi_launch.py \
        --files "${ONEFLOW_TEST_DIR}/**/test_*.py" \
        -n 4 \
        --group_size $device_num \
        --device_num 4 \
        --verbose \
        --auto_cuda_visible_devices \
        -m oneflow.distributed.launch \
        --nproc_per_node $device_num --nnodes=2 --node_rank=$NODE_RANK --master_addr $_MASTER_ADDR \
        -m pytest --max-worker-restart=0 -x --durations=50 --capture=sys -p no:cacheprovider -p no:randomly --ignore=log
done
