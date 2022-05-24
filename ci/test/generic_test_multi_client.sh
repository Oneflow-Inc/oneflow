#!/bin/bash
set -xe

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
ONEFLOW_TEST_DIR=${ONEFLOW_TEST_DIR:-"$PWD/python/oneflow/test/modules"}
ONEFLOW_TEST_TASKS_PER_GPU=${ONEFLOW_TEST_TASKS_PER_GPU:-"4"}

if [ -z "$ONEFLOW_TEST_CPU_ONLY" ]
then
    gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    for ((i=0;i<gpu_num;i++)); do
        for ((j=0;j<ONEFLOW_TEST_TASKS_PER_GPU;j++)); do
            parallel_spec="$parallel_spec --tx popen//env:CUDA_VISIBLE_DEVICES=${i}"
        done
    done
    multi_launch_device_num=${gpu_num}
else
    parallel_spec="-n auto"
    multi_launch_device_num=8
fi

unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

export ONEFLOW_TEST_DEVICE_NUM=1

COMMON_PYTEST_ARGS="-p no:warnings -p no:randomly -p no:cacheprovider --max-worker-restart=0 -x --durations=50 --capture=sys --ignore=log"
time python3 -m pytest ${COMMON_PYTEST_ARGS} --dist loadfile ${parallel_spec} ${ONEFLOW_TEST_DIR}
if [[ "$(python3 -c 'import oneflow.sysconfig;print(oneflow.sysconfig.has_rpc_backend_grpc())')" == *"True"* ]]; then
    export ONEFLOW_TEST_DEVICE_NUM=2
    time python3 ${src_dir}/ci/test/multi_launch.py \
        --files "${ONEFLOW_TEST_DIR}/**/test_*.py" \
        --master_port 29500 \
        --master_port 29501 \
        --master_port 29502 \
        --master_port 29503 \
        -n master_port \
        --group_size 2 \
        --auto_cuda_visible_devices \
        --device_num $multi_launch_device_num \
        -m oneflow.distributed.launch --nproc_per_node 2 -m pytest ${COMMON_PYTEST_ARGS}

    export ONEFLOW_TEST_DEVICE_NUM=4
    time python3 ${src_dir}/ci/test/multi_launch.py \
        --files "${ONEFLOW_TEST_DIR}/**/test_*.py" \
        -n 4 \
        --group_size 4 \
        --device_num $multi_launch_device_num \
        --auto_cuda_visible_devices \
        -m oneflow.distributed.launch --nproc_per_node 4 -m pytest ${COMMON_PYTEST_ARGS}
else
    python3 -c 'import oneflow.sysconfig;assert(oneflow.sysconfig.has_rpc_backend_grpc() == False)'
fi
