set -e
set -x
ci_tmp=$1

chmod +x $ci_tmp/oneflow_testexe
$ci_tmp/oneflow_testexe

pip3 install --user $ci_tmp/*.whl
installed_path=$(python3 -c "import oneflow; print(oneflow.__path__[0])")
test_dir="/test_dir"
cp -r $ci_tmp/oneflow/python/test $test_dir
python3 $test_dir/ops/1node_test.py
python3 $test_dir/models/1node_test.py
