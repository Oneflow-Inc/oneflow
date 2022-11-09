set -ex
src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"$PWD/build-docs"}
rm -rf $test_tmp_dir
cp -r docs ${test_tmp_dir}
cd ${test_tmp_dir}

make html SPHINXOPTS="-W --keep-going"
