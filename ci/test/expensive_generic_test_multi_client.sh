#!/bin/bash
set -xe

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
ONEFLOW_TEST_DIR=${ONEFLOW_TEST_DIR:-"$PWD/python/oneflow/test/modules"}

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

COMMON_PYTEST_ARGS="--max-worker-restart=0 -x --durations=50 --capture=sys"
python3 -m pytest ${COMMON_PYTEST_ARGS} --failed-first --dist loadfile ${parallel_spec} ${PWD}
if [[ "$(python3 -c 'import oneflow.sysconfig;print(oneflow.sysconfig.has_rpc_backend_grpc())')" == *"True"* ]]; then
    export ONEFLOW_TEST_DEVICE_NUM=2
    python3 -m oneflow.distributed.launch --nproc_per_node 2 -m pytest ${COMMON_PYTEST_ARGS} ${PWD}

    export ONEFLOW_TEST_DEVICE_NUM=4
    python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest ${COMMON_PYTEST_ARGS} ${PWD}
else
    python3 -c 'import oneflow.sysconfig;assert(oneflow.sysconfig.has_rpc_backend_grpc() == False)'
fi
