#!/bin/bash

set -xeu

export PYTHONUNBUFFERED=1

function do_test() {
    ONEFLOW_TEST_NODE_NUM=2 ONEFLOW_TEST_DEVICE_NUM=$device_num python3 -m oneflow.distributed.launch --nproc_per_node $device_num --nnodes=2 --node_rank=$NODE_RANK --master_addr $_MASTER_ADDR -m unittest discover ${ONEFLOW_TEST_DIR} --failfast --verbose
}

for device_num in 1 2 4
do
    # use a invalid ibverbs lib to test if falling back to epoll works
    do_test
    ONEFLOW_LIBIBVERBS_PATH=invalid_lib do_test
done
