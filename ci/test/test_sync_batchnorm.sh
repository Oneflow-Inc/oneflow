#!/usr/bin/env bash

set -ex
cd $ONEFLOW_TEST_DIR

export ONEFLOW_TEST_DEVICE_NUM=2
python3 -m oneflow.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 test_sync_batchnorm.py
python3 -m oneflow.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 test_sync_batchnorm_nhwc.py
