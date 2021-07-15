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

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export ONEFLOW_TEST_DEVICE_NUM=1
python3 $src_dir/ci/test/parallel_run.py \
    --gpu_num=${gpu_num} \
    --dir=${PWD} \
    --timeout=1 \
    --verbose \
    --chunk=1

export ONEFLOW_TEST_DEVICE_NUM=2
for f in test_*.py
do
    python3 -m oneflow.distributed.launch --nproc_per_node 2 $f
done

export ONEFLOW_TEST_DEVICE_NUM=4
for f in test_*.py
do
    python3 -m oneflow.distributed.launch --nproc_per_node 4 $f
done
