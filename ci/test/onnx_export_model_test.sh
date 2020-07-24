#!/bin/bash

set -xe

pip3 install --user ci_tmp/*.whl

test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"/test_tmp_dir"}
rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
ls oneflow/python/test/onnx
cp -r oneflow/python/test/onnx/ $test_tmp_dir
cd $test_tmp_dir
ls
python3 test_model.py
