#!/bin/bash
set -e
MOCK_TORCH=$PWD/python/oneflow/test/misc/mock_example.py

same_or_exit() {
    if [[ "$(python3 $MOCK_TORCH)" != *"$1"* ]]; then
        exit 1
    fi
}

# generate pytorch file
python3 -c "import torch; torch.save(torch.ones(1), 'test.pt')"

eval $(python3 -m oneflow.mock_torch) # test call to python module, default argument is enable
same_or_exit "True"

# test load pytorch file with mock torch enabled
python3 -c """
import torch
x = torch.load('test.pt')
assert torch.equal(x, torch.ones(1))
import torch.nn
assert 'oneflow/nn/__init__.py' in torch.nn.__file__
"""

# testing import
python3 -c 'import torch; torch.randn(2,3)'
python3 -c 'import torch.nn; torch.nn.Graph'
python3 -c 'import torch.version; torch.version.__version__'
python3 -c 'from torch import *; randn(2,3)'
python3 -c 'from torch.nn import *; Graph'
python3 -c 'from torch.sbp import *; sbp'
python3 -c 'from torch import nn; nn.Graph'
python3 -c 'from torch.version import __version__'
python3 -c 'import torch; torch.not_exist' 2>&1 >/dev/null | grep -q 'AttributeError'
python3 -c 'import torch.not_exist' 2>&1 >/dev/null | grep -q 'ModuleNotFoundError'

eval $(python3 -m oneflow.mock_torch disable)
same_or_exit "False"
eval $(python3 -m oneflow.mock_torch enable)
same_or_exit "True"
eval $(python3 -m oneflow.mock_torch disable) # recover
same_or_exit "False"
eval $(oneflow-mock-torch) # test scripts
same_or_exit "True"
eval $(oneflow-mock-torch disable)
same_or_exit "False"
eval $(oneflow-mock-torch enable)
same_or_exit "True"
eval $(oneflow-mock-torch disable)
same_or_exit "False"

# test load pytorch file with mock torch disabled
python3 -c "import oneflow as flow; x = flow.load('test.pt'); assert flow.equal(x, flow.ones(1))"

rm test.pt

eval $(python3 -m oneflow.mock_torch --lazy --verbose)
python3 -c "import torch.not_exist" | grep -q 'dummy object'
