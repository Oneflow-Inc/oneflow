#!/bin/bash

set -xeu

export PYTHONUNBUFFERED=1

for device_num in 1 2 4
do
    ONEFLOW_TEST_NODE_NUM=2 ONEFLOW_TEST_DEVICE_NUM=$device_num python3 -m oneflow.distributed.launch --nproc_per_node $device_num --nnodes=2 --node_rank=$NODE_RANK --master_addr $_MASTER_ADDR -m unittest discover ${ONEFLOW_TEST_DIR} --failfast --verbose
done
