#!/usr/bin/env bash

set -ex

cd $ONEFLOW_TEST_DIR

python3 -m oneflow.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 test_sync_batchnorm.py
