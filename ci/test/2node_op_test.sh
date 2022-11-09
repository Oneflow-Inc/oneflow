#!/bin/bash
set -xe

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"./test_tmp_dir"}


rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
chmod -R o+w $test_tmp_dir
cp -r $src_dir/python/oneflow/compatible/single_client/test $test_tmp_dir
cd $test_tmp_dir

ONEFLOW_TEST_DEVICE_NUM=1 python3 test/ops/test_assign.py --failfast --verbose
ONEFLOW_TEST_DEVICE_NUM=1 python3 test/ops/test_two_node_boxing.py --failfast --verbose

for device_num in 1 2 4
do
    ONEFLOW_TEST_ENABLE_INIT_BY_HOST_LIST=1 ONEFLOW_TEST_DEVICE_NUM=$device_num python3 -m unittest discover test/ops --failfast --verbose
    # use a invalid ibverbs lib to test if falling back to epoll works
    ONEFLOW_TEST_ENABLE_INIT_BY_HOST_LIST=1 ONEFLOW_TEST_DEVICE_NUM=$device_num ONEFLOW_LIBIBVERBS_PATH=invalid_lib python3 -m unittest discover test/ops --failfast --verbose
done
