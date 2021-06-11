#!/bin/bash
set -xe
export PYTHONUNBUFFERED=1
src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}

cd `python3 -c "import oneflow; print(oneflow.__path__[0])"`
cd ..

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python3 $src_dir/ci/test/parallel_run.py \
    --gpu_num=${gpu_num} \
    --dir=${PWD} \
    --timeout=1 \
    --verbose \
    --chunk=1 \
    --doctest
