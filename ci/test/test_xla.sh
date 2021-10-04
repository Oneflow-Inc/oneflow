set -ex

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"./test_tmp_dir"}

rm -rf $test_tmp_dir
cp -r $src_dir/python/oneflow/compatible/single_client/test/xrt $test_tmp_dir
cd $test_tmp_dir
python3 -c "import oneflow.compatible.single_client as flow; assert flow.sysconfig.with_xla()"
gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python3 $src_dir/ci/test/parallel_run.py \
    --gpu_num=${gpu_num} \
    --dir=${PWD} \
    --timeout=1 \
    --verbose \
    --chunk=1
