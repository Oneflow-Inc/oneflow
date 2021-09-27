set -ex

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"./test_tmp_dir"}

rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
cp -r $src_dir/python/oneflow/compatible/single_client/test/xrt $test_tmp_dir
cd $test_tmp_dir
python3 -c "import oneflow.compatible.single_client as flow; assert flow.sysconfig.with_xla()"
for f in $src_dir/python/oneflow/compatible/single_client/test/xrt/*.py; do python3 "$f"; done
