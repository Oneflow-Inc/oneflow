#!/bin/bash
set -xe

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
wheel_path=${ONEFLOW_WHEEL_PATH:-"ci_tmp/*.whl"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"/test_tmp_dir"}

if [ -f "$wheel_path" ]; then
    pip3 install --user "$wheel_path"
elif [ -d "$src_dir" ]; then
    pip3 install -e "$src_dir" --user
else
    echo "wheel not found: $wheel_path, src dir not found: $src_dir, continue anyway..."
fi


rm -rf $test_tmp_dir
cp -r $src_dir/oneflow/python/test $test_tmp_dir
cd $test_tmp_dir

python3 ops/1node_test.py
