#!/bin/bash
set -xe

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_dir=${ONEFLOW_TEST_DIR:-"$PWD/python/oneflow/test/ops"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"./test_tmp_dir"}
export ONEFLOW_TEST_UTILS_DIR=$src_dir/python/oneflow/test_utils

rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
cp -r $test_dir $test_tmp_dir
cd ${test_tmp_dir}/$(basename $test_dir)

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export ONEFLOW_TEST_DEVICE_NUM=1
python3 $src_dir/ci/test/parallel_run.py \
    --gpu_num=${gpu_num} \
    --dir=${PWD} \
    --timeout=1 \
    --verbose \
    --chunk=1

export ONEFLOW_TEST_DEVICE_NUM=2
python3 -m unittest discover ${PWD} --failfast --verbose

export ONEFLOW_TEST_DEVICE_NUM=4
python3 -m unittest discover ${PWD} --failfast --verbose
