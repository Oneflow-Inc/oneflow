#!/bin/bash
set -xe

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
ONEFLOW_TEST_DIR=${ONEFLOW_TEST_DIR:-"$PWD/python/oneflow/test/modules"}

cd $ONEFLOW_TEST_DIR

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export ONEFLOW_TEST_DEVICE_NUM=1
python3 -m pytest -n 4  ${PWD} --verbose --durations=50 -x
if [[ "$(python3 -c 'import oneflow.sysconfig;print(oneflow.sysconfig.has_rpc_backend_grpc())')" == *"True"* ]]; then
    export ONEFLOW_TEST_DEVICE_NUM=2
    python3 -m oneflow.distributed.launch --nproc_per_node 2 -m unittest discover ${PWD} --failfast --verbose

    export ONEFLOW_TEST_DEVICE_NUM=4
    python3 -m oneflow.distributed.launch --nproc_per_node 4 -m unittest discover ${PWD} --failfast --verbose
else
    python3 -c 'import oneflow.sysconfig;assert(oneflow.sysconfig.has_rpc_backend_grpc() == False)'
fi
