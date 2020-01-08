pip3 install *.whl
installed_path=$(python3 -c "import oneflow; print(oneflow.__path__[0])")
python3 oneflow/python/test/ops/1node_test.py
python3 oneflow/python/test/models/1node_test.py