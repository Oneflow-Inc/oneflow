#!/bin/bash
set -xe

export PYTHONUNBUFFERED=1

export ONEFLOW_TEST_UTILS_DIR=$src_dir/python/oneflow/test_utils

export ONEFLOW_TEST_DEVICE_NUM=8
echo "test_neq_device_process_num begin"
python3 -m oneflow.distributed.launch --nproc_per_node 8 -m unittest discover ${PWD}/python/oneflow/test/graph/test_neq_device_process_num.py --failfast --verbose
echo "test_neq_device_process_num end"
