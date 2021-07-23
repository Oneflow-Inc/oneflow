#!/bin/bash
set -xe

cp -r oneflow/python/test /test_dir
cd /test_dir

python3 models/eager_1node_test.py
