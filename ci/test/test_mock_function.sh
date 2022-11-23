#!/bin/bash
set -e
MOCK_UNITTEST=$PWD/python/oneflow/test/misc/test_mock_scope.py

python3 $MOCK_UNITTEST --failfast --verbose
# testing import *
python3 -c "
import oneflow
import oneflow.nn
import oneflow.mock_torch as mock; mock.enable();
from torch.sbp import *; assert(sbp == oneflow.sbp.sbp);
from torch import *; assert(randn == oneflow.randn);
from torch.nn import *; assert(Graph == oneflow.nn.Graph);
mock.disable();
from torch import *; assert(randn != oneflow.randn);
from torch.nn import *; assert(Graph != oneflow.nn.Graph);
"
