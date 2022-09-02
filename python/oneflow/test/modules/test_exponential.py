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


def _test_exponential(test_case, shape):
    from scipy.stats import kstest

    lambd = random(low=0).to(float)
    tensor = torch.randn(shape, dtype=torch.float32)
    tensor.exponential_(lambd=lambd)
    pvalue = kstest(
        tensor.oneflow.flatten().numpy(), tensor.pytorch.flatten().numpy()
    ).pvalue
    test_case.assertTrue(pvalue > 0.05)


def _test_exponential_with_generator(test_case, shape):
    generator = flow.Generator()
    generator.manual_seed(0)
    x = flow.tensor(
        np.random.rand(*shape), dtype=flow.float32, device=flow.device("cpu")
    )
    y_1 = flow.exponential(x, generator=generator)
    generator.manual_seed(0)
    y_2 = flow.exponential(x, generator=generator)
    test_case.assertTrue(np.allclose(y_1.numpy(), y_2.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestExponential(flow.unittest.TestCase):
    @unittest.skip("sometimes p-value maybe less than 0.05")
    def test_exponential(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_functions"] = [_test_exponential]
        arg_dict["shape"] = [(20, 30), (20, 30, 40), (20, 30, 40, 50)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_exponential_with_generator(test_case):
        for shape in [(2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            _test_exponential_with_generator(test_case, shape)


if __name__ == "__main__":
    unittest.main()
