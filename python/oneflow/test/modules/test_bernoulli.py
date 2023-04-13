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


def _test_bernoulli(test_case, shape, p, dtype):
    input_arr = np.ones(shape)
    x = flow.tensor(input_arr, dtype=flow.float32, device=flow.device("cpu"))
    if p is None:
        y = flow.bernoulli(x, dtype=dtype)
    else:
        y = flow.bernoulli(x, p=p, dtype=dtype)
    test_case.assertTrue(y.dtype == dtype)
    if p == 1 or p is None:
        test_case.assertTrue(np.allclose(y.numpy(), x.numpy()))
    elif p == 0:
        test_case.assertTrue(np.allclose(y.numpy(), np.zeros(shape)))


def _test_bernoulli_with_generator(test_case, shape):
    generator = flow.Generator()
    generator.manual_seed(0)
    x = flow.tensor(
        np.random.rand(*shape), dtype=flow.float32, device=flow.device("cpu")
    )
    y_1 = flow.bernoulli(x, generator=generator)
    generator.manual_seed(0)
    y_2 = flow.bernoulli(x, generator=generator)
    test_case.assertTrue(np.allclose(y_1.numpy(), y_2.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestBernoulli(flow.unittest.TestCase):
    def test_bernoulli(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_functions"] = [_test_bernoulli]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["p"] = [None, 0, 1]
        arg_dict["dtype"] = [flow.float32, flow.int64]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skip("bernoulli has bug")
    @autotest(auto_backward=False)
    def test_flow_bernoulli_with_random_data(test_case):
        input = random_tensor(ndim=1).to("cpu")
        return torch.bernoulli(input)

    """
    @profile(torch.bernoulli) 
    def profile_bernoulli(test_case):
        torch.bernoulli(torch.ones(3, 3))
        torch.bernoulli(torch.zeros(3, 3))
    """


if __name__ == "__main__":
    unittest.main()
