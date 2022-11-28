#!/bin/bash
set -xe
export PYTHONUNBUFFERED=1
src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"./test_tmp_dir"}

mkdir -p ${test_tmp_dir}
cd ${test_tmp_dir}
python3 -c 'import oneflow; f=open("oneflow_path.txt", "w"); f.write(oneflow.__path__[0])'

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python3 $src_dir/ci/test/parallel_run.py \
    --gpu_num=${gpu_num} \
    --dir=$(cat oneflow_path.txt) \
    --timeout=1 \
    --verbose \
    --chunk=1 \
    --doctest
