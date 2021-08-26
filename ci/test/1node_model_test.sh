#!/bin/bash
set -xe

cp -r python/oneflow/compatible/single_client/test /test_dir
cd /test_dir

python3 models/1node_test.py
