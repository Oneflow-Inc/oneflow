"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict

import numpy as np

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util.torch_flow_dual_object import autotest


@autotest(n=1, check_graph=False)
def _test_search_sorted(test_case, placement, sbp):
    sorted_sequence = random_tensor(ndim=2, dim0=2, dim1=3).to_global(placement, sbp)
    values = random_tensor(ndim=2, dim0=2).to_global(placement, sbp)
    right = oneof(True, False)
    y = torch.searchsorted(
        sorted_sequence,
        values,
        out_int32=oneof(True, False),
        right=right,
    )
    return y


class TestSearchSorted_Global(flow.unittest.TestCase):
    @globaltest
    def test_search_sorted(test_case):
        placement = torch.placement("cuda", ranks=[0, 1])
        sbp = torch.sbp.split(0)
        _test_search_sorted(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
