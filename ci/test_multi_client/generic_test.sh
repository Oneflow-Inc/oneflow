#!/bin/bash
set -xe

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_dir=${ONEFLOW_TEST_DIR:-"$PWD/oneflow/python/test/ops"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"./test_tmp_dir"}


rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
cp -r $test_dir $test_tmp_dir
cd ${test_tmp_dir}/$(basename $test_dir)

export ONEFLOW_TEST_DEVICE_NUM=2
for f in test/modules/test_*.py
do
    python3 -m oneflow.distributed.launch --nproc_per_node 2 $f
done

export ONEFLOW_TEST_DEVICE_NUM=4
for f in test/modules/test_*.py
do
    python3 -m oneflow.distributed.launch --nproc_per_node 4 $f
done
