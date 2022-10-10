#!/bin/bash

oneflow_py_dir="$PWD/python"
simple_torch_file=$oneflow_py_dir/oneflow/test/misc/test_mock_torch.py
# python3 -m pip install torch

cd $oneflow_py_dir
# python3 setup.py install --user
eval $(oneflow-mock-torch enable)
if [[ "$(python3 $simple_torch_file)" != *"True"* ]]; then
exit 1
fi
eval $(oneflow-mock-torch disable)
if [[ "$(python3 $simple_torch_file)" != *"False"* ]]; then
exit 1
fi
