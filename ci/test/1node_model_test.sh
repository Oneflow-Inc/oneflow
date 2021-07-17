#!/bin/bash
set -xe

cp -r oneflow/compatible_single_client_python/test /test_dir
cd /test_dir

python3 models/1node_test.py
