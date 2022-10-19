#!/bin/bash
set -e
MOCK_UNITTEST=$PWD/python/oneflow/test/misc/test_mock.py

python3 $MOCK_UNITTEST --failfast --verbose
# mocking won't work because torch is already imported
python3 -c "import torch; from oneflow.mock_torch import mock;
mock(); assert(torch.__package__ == 'torch')"
# testing import *
python3 -c "
import oneflow
import oneflow.nn
from oneflow.mock_torch import mock; mock();
from torch.sbp import *; assert(sbp == oneflow.sbp.sbp);
from torch import *; assert(randn == oneflow.randn);
from torch.nn import *; assert(Graph == oneflow.nn.Graph)"
