#!/bin/bash
set -xe

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"/test_tmp_dir"}


rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
chmod -R o+w $test_tmp_dir
cp -r $src_dir/oneflow/python/test $test_tmp_dir
cd $test_tmp_dir


for device_num in 1 2 4
do
    ONEFLOW_TEST_ENABLE_INIT_BY_HOST_LIST=1 ONEFLOW_TEST_DEVICE_NUM=$device_num python3 test/ops/test_quantization_aware_training.py --failfast --verbose
done
