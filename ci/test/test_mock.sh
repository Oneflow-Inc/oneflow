#!/bin/bash
set -e
MOCK_TORCH=$PWD/python/oneflow/test/misc/test_mock_simple.py
MOCK_UNITTEST=$PWD/python/oneflow/test/misc/test_mock.py

same_or_exit() {
    if [[ "$(python3 $MOCK_TORCH)" != *"$1"* ]]; then
        exit 1
    fi
}
eval $(python3 -m oneflow.mock_torch) # default argument is enable
same_or_exit "True"

# testing import
python3 -c 'import torch; torch.randn(2,3)'
python3 -c 'import torch.nn; torch.nn.Graph' 
python3 -c 'import torch.version; torch.version.__version__' 
python3 -c 'from torch import *; randn(2,3)' 
python3 -c 'from torch.nn import *; Graph' 
python3 -c 'from torch.version import *; __version__' 
python3 -c 'from torch import nn; nn.Graph' 
python3 -c 'from torch.version import __version__' 
! (python3 -c 'import torch; torch.no_exist' 2>&1 >/dev/null | grep -q 'NotImplementedError')

eval $(python3 -m oneflow.mock_torch disable)
same_or_exit "False"
eval $(python3 -m oneflow.mock_torch enable)
same_or_exit "True"
eval $(python3 -m oneflow.mock_torch disable) # recover
same_or_exit "False"
eval $(oneflow-mock-torch) 
same_or_exit "True"
eval $(oneflow-mock-torch disable)
same_or_exit "False"
eval $(oneflow-mock-torch enable)
same_or_exit "True"
eval $(oneflow-mock-torch disable)
same_or_exit "False"
python3 $MOCK_UNITTEST --failfast --verbose
# mocking won't work because torch is already imported
python3 -c "import torch; from oneflow.mock_torch import mock; 
mock(); assert(torch.__package__ == 'torch')"
# testing import *
python3 -c "
import oneflow
from oneflow.mock_torch import mock; mock();
from torch import *;
from torch.nn import *;
from torch.version import *"
