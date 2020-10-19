#!/bin/bash
set -xe

cp -r oneflow/python/test /test_dir
cd /test_dir

python3 models/1node_test.py
