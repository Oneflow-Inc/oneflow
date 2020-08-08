#!/bin/bash

set -xe

test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"/test_tmp_dir"}
rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
cp -r oneflow/python/test/onnx/* $test_tmp_dir
cd $test_tmp_dir
python3 test_model.py
python3 test_node.py test_conv test_pooling
