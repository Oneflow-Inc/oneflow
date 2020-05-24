#!/bin/bash
set -xe

pip3 install --user ci_tmp/*.whl

cp -r oneflow/python/test /test_dir
cd /test_dir
export ENABLE_USER_OP=True
python3 models/1node_test.py
python3 models/1node_test.py --enable_auto_mixed_precision=True
