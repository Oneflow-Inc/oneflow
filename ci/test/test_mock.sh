#!/bin/bash

MOCK_TORCH=$PWD/python/oneflow/test/misc/test_mock_torch.py

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
if [[ "$(python3 $MOCK_TORCH)" != *"False"* ]]; then
    exit 1
fi
