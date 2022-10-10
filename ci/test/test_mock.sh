#!/bin/bash

ONEFLOW_PY_DIR="$PWD/python"
MOCK_TORCH=$ONEFLOW_PY_DIR/oneflow/test/misc/test_mock_torch.py

cd $ONEFLOW_PY_DIR
python3 setup.py install --user
eval $(oneflow-mock-torch) # default argument is enable
if [[ "$(python3 $MOCK_TORCH)" != *"True"* ]]; then
    exit 1
fi
eval $(oneflow-mock-torch disable)
if [[ "$(python3 $MOCK_TORCH)" != *"False"* ]]; then
    exit 1
fi
eval $(oneflow-mock-torch enable)
if [[ "$(python3 $MOCK_TORCH)" != *"True"* ]]; then
    exit 1
fi
eval $(oneflow-mock-torch disable) # recover
