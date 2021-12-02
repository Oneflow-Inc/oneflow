#!/bin/bash
set -xe

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"./test_tmp_dir"}
export ONEFLOW_TEST_UTILS_DIR=$src_dir/python/oneflow/test_utils

rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
test_neq_device_process_file_path=${PWD}/python/oneflow/test/graph/test_neq_device_process_num.py
if [ -e $test_neq_device_process_file_path ] && [ "$(python3 -c 'import oneflow.sysconfig;print(oneflow.sysconfig.has_rpc_backend_grpc())')" == *"True"* ]
then
    cp test_neq_device_process_file_path $test_tmp_dir
    cd ${test_tmp_dir}
    python3 -m oneflow.distributed.launch --nproc_per_node 8 -m unittest ./test_neq_device_process_num.py --failfast --verbose
    echo "nn.Graph test process num > device num done."
fi
