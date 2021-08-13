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


def random_expand(x, ndim, expand_size):
    dim_size = [1,] * ndim
    random_index = random(0, ndim).to(int).value()
    dim_size[random_index] = expand_size
    return x.expand(*dim_size)


@flow.unittest.skip_unless_1n1d()
class TestExpand(flow.unittest.TestCase):
    @autotest()
    def test_flow_tensor_expand_with_random_data(test_case):
        random_expand_size = random(1, 6).to(int).value()
        x = random_pytorch_tensor(ndim=5, dim0=1, dim1=1, dim2=1, dim3=1, dim4=1)
        return random_expand(x, ndim=5, expand_size=random_expand_size)


if __name__ == "__main__":
    unittest.main()
