#!/bin/bash
set -xe

chmod +x ./ci_tmp/oneflow_testexe
./ci_tmp/oneflow_testexe

pip3 install --user ci_tmp/*.whl

cp -r oneflow/python/test /test_dir
cd /test_dir
export ENABLE_USER_OP=True
python3 ops/1node_test.py
