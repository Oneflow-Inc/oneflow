#!/bin/bash

set -xeu

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_dir=${ONEFLOW_TEST_DIR:-"$PWD/python/oneflow/test/modules"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"./test_tmp_dir"}
export ONEFLOW_TEST_UTILS_DIR=$src_dir/python/oneflow/test_utils


rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
cp -r $test_dir $test_tmp_dir
cd ${test_tmp_dir}/$(basename $test_dir)

for device_num in 1 2 4
do
    ONEFLOW_TEST_NODE_NUM=2 ONEFLOW_TEST_DEVICE_NUM=$device_num python3 -m oneflow.distributed.launch --nproc_per_node $device_num --nnodes=2 --node_rank=$NODE_RANK --master_addr $_MASTER_ADDR -m unittest discover ${PWD} --failfast --verbose
    # use a invalid ibverbs lib to test if falling back to epoll works
    ONEFLOW_TEST_NODE_NUM=2 ONEFLOW_TEST_DEVICE_NUM=$device_num ONEFLOW_LIBIBVERBS_PATH=invalid_lib python3 -m oneflow.distributed.launch --nproc_per_node $device_num --nnodes=2 --node_rank=$NODE_RANK --master_addr $_MASTER_ADDR -m unittest discover ${PWD} --failfast --verbose
done
