#!/bin/bash
set -xe

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
ONEFLOW_TEST_DIR=${ONEFLOW_TEST_DIR:-"$PWD/oneflow/ir/test"}

cd $ONEFLOW_TEST_DIR

if [ -z "$ONEFLOW_TEST_CPU_ONLY" ]
then
    gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    for ((i=0;i<gpu_num;i++)); do
        parallel_spec="$parallel_spec --tx popen//env:CUDA_VISIBLE_DEVICES=${i}"
    done
else
    parallel_spec="-n auto"
fi

unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

export ONEFLOW_TEST_DEVICE_NUM=1

COMMON_PYTEST_ARGS="--max-worker-restart=0 --durations=50 --ignore=OneFlow/cuda_code_gen --ignore=OneFlow/psig/test_2nd_basic_parse.py"
python3 -m pytest ${COMMON_PYTEST_ARGS} --failed-first --dist loadfile ${parallel_spec} ${PWD}
