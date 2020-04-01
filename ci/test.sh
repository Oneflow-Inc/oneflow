set -e
set -x

chmod +x ./ci_tmp/oneflow_testexe
./ci_tmp/oneflow_testexe

pip3 install --user ci_tmp/*.whl --user
test_dir="/test_dir"
cp -r oneflow/python/test $test_dir
python3 $test_dir/ops/1node_test.py
python3 $test_dir/models/1node_test.py
