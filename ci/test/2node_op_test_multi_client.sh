#!/bin/bash

set -xeu

export PYTHONUNBUFFERED=1

ONEFLOW_CI_DEVICE_NUMS=${ONEFLOW_CI_DEVICE_NUMS:-"1 2 4"}

for device_num in ${ONEFLOW_CI_DEVICE_NUMS}
do
    ONEFLOW_TEST_NODE_NUM=2 ONEFLOW_TEST_DEVICE_NUM=$device_num python3 -m oneflow.distributed.launch --nproc_per_node $device_num --nnodes=2 --node_rank=$NODE_RANK --master_addr $_MASTER_ADDR -m pytest --max-worker-restart=0 -x --durations=50 --capture=sys ${ONEFLOW_TEST_DIR}
done
