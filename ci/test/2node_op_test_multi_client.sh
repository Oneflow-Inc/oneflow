#!/bin/bash

set -xeu

echo "HTTP_PROXY:  ${HTTP_PROXY}"
echo "HTTPS_PROXY: ${HTTPS_PROXY}"
echo "http_proxy:  ${http_proxy}"
echo "https_proxy: ${https_proxy}"
export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_dir=${ONEFLOW_TEST_DIR:-"$PWD/python/oneflow/test/modules"}
cd ${test_dir}

for device_num in 1 2 4
do
    ONEFLOW_TEST_NODE_NUM=2 ONEFLOW_TEST_DEVICE_NUM=$device_num python3 -m oneflow.distributed.launch --nproc_per_node $device_num --nnodes=2 --node_rank=$NODE_RANK --master_addr $_MASTER_ADDR -m unittest discover ${PWD} --failfast --verbose
    # use a invalid ibverbs lib to test if falling back to epoll works
    ONEFLOW_TEST_NODE_NUM=2 ONEFLOW_TEST_DEVICE_NUM=$device_num ONEFLOW_LIBIBVERBS_PATH=invalid_lib python3 -m oneflow.distributed.launch --nproc_per_node $device_num --nnodes=2 --node_rank=$NODE_RANK --master_addr $_MASTER_ADDR -m unittest discover ${PWD} --failfast --verbose
done
