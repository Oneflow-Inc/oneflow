#!/bin/bash

MOCK_TORCH=$PWD/python/oneflow/test/misc/test_mock_torch.py

eval $(python3 -m oneflow.mock_torch) # default argument is enable
if [[ "$(python3 $MOCK_TORCH)" != *"True"* ]]; then
    exit 1
fi
# testing import
if
    python3 -c 'import torch; torch.randn(2,3)' &&\
    python3 -c 'import torch.nn; torch.nn.Graph' &&\
    python3 -c 'import torch.version; torch.version.__version__' &&\
    python3 -c 'from torch import *' &&\
    python3 -c 'from torch.nn import *' &&\
    python3 -c 'from torch.version import *' &&\
    python3 -c 'from torch import nn' &&\
    python3 -c 'from torch.version import __version__' &&\
    ! (python3 -c 'import torch; torch.no_exist' 2>&1 >/dev/null | grep -q 'NotImplementedError')
then
    exit 1
fi
eval $(python3 -m oneflow.mock_torch disable)
if [[ "$(python3 $MOCK_TORCH)" != *"False"* ]]; then
    exit 1
fi
eval $(python3 -m oneflow.mock_torch enable)
if [[ "$(python3 $MOCK_TORCH)" != *"True"* ]]; then
    exit 1
fi
eval $(python3 -m oneflow.mock_torch disable) # recover
if [[ "$(python3 $MOCK_TORCH)" != *"False"* ]]; then
    exit 1
fi
