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
import math as math

import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList


def _test_lgamma(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.lgamma(input)
    print(of_out.numpy())
    np_out = np.fromiter((map(math.lgamma, np.nditer(input.numpy()))), dtype = np.float32)
    # test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

# def _test_lgamma_backward(test_case, shape, device):
#     input = flow.Tensor(np.random.randn(*shape), requires_grad=True, device=flow.device(device))

#     of_out = flow.lgamma(input)
#     of_out.backward()
#     test_case.assertTrue(np.allclose(x.grad.numpy(), np.zeros(shape), 1e-4, 1e-4))

#     input = flow.Tensor(np.random.randn(*shape), requires_grad=True, device=flow.device(device))
#     of_out = input.ceil()
#     of_out.backward()
#     test_case.assertTrue(np.allclose(x.grad.numpy(), np.zeros(shape), 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLess(flow.unittest.TestCase):
    def test_lgamma(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_lgamma,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
