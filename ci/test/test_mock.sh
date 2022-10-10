#!/bin/bash

MOCK_TORCH=$PWD/python/oneflow/test/misc/test_mock_torch.py

eval $(oneflow-mock-torch) # default argument is enable
if [[ "$(python3 $MOCK_TORCH)" != *"True"* ]]; then
    exit 1
fi
# testing import
python3 -c 'import torch; torch.randn(2,3)'
python3 -c 'import torch.nn; torch.nn.Graph'
python3 -c 'import torch; torch.no_exist' 2>&1 >/dev/null | grep 'NotImplementedError'

eval $(oneflow-mock-torch disable)
if [[ "$(python3 $MOCK_TORCH)" != *"False"* ]]; then
    exit 1
fi
eval $(oneflow-mock-torch enable)
if [[ "$(python3 $MOCK_TORCH)" != *"True"* ]]; then
    exit 1
fi
eval $(oneflow-mock-torch disable) # recover
if [[ "$(python3 $MOCK_TORCH)" != *"False"* ]]; then
    exit 1
fi
