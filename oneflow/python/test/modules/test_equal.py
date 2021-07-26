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

import oneflow as flow
from test_util import GenArgList
from automated_test_util import *


def get_random_data_with_same_size():
    ndim = random(1, 6)
    dim0 = random(1, 6)
    dim1 = random(1, 6)
    dim2 = random(1, 6)
    dim3 = random(1, 6)
    dim4 = random(1, 6)
    x = random_tensor(ndim=ndim, dim0=dim0, dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4)
    y = random_tensor(ndim=ndim, dim0=dim0, dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4)
    return x, y

@flow.unittest.skip_unless_1n1d()
class TestEqual(flow.unittest.TestCase):  

    @autotest()
    def test_flow_equal_with_random_data(test_case):
        x, y = get_random_data_with_same_size()
        z = torch.equal(x, y)
        return z


if __name__ == "__main__":
    unittest.main()
