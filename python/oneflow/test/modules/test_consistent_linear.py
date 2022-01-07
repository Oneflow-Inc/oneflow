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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestAddModule(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_linear_with_random_data(test_case):
        try:
            input_size = random()
            placement = random_cpu_placement()
            sbp = random_sbp(placement, max_dim=1)
            m = torch.nn.Linear(
                in_features=input_size, out_features=random(), bias=random() | nothing()
            )
            m.train(random())
            # m = m.to_consistent(placement=placement, sbp=sbp)
            m.weight = torch.nn.Parameter(
                m.weight.to_consistent(placement=placement, sbp=sbp)
            )
            if m.bias is not None:
                m.bias = torch.nn.Parameter(
                    m.bias.to_consistent(placement=placement, sbp=sbp)
                )
            x = random_pytorch_tensor(ndim=2, dim1=input_size).to_consistent(
                placement=placement, sbp=sbp
            )
            y = m(x)
            return y
        except Exception as e:
            assert "can't find available sbp signature." in str(e)


if __name__ == "__main__":
    unittest.main()
