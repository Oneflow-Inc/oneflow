set -x
pwd=$1
pip3 install --user $pwd/*.whl
installed_path=$(python3 -c "import oneflow; print(oneflow.__path__[0])")
test_dir="/test_dir"
cp -r $pwd/oneflow/python/test $test_dir
touch $pwd/test_result.txt
python3 $test_dir/ops/1node_test.py >> $pwd/test_result.txt
python3 $test_dir/models/1node_test.py >> $pwd/test_result.txt
chmod 660 $pwd/test_result.txt
