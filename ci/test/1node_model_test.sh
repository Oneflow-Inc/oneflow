#!/bin/bash
set -xe

cp -r oneflow/compatible/single_client/python/test /test_dir
cd /test_dir

python3 models/1node_test.py
