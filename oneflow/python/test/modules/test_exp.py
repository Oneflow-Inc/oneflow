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

import numpy as np
from collections import OrderedDict
import oneflow.experimental as flow
from test_util import GenArgList

def _test_exp(test_case, device):
    input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device))
    of_out = flow.exp(input)
    np_out = np.exp(input.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

def _test_tensor_exp(test_case, device):
    input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device))
    of_out = input.exp()
    np_out = np.exp(input.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

def _test_exp_backward(test_case, device):
    input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device), requires_grad=True)
    of_out = flow.exp(input)
    np_grad = of_out.numpy()
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, rtol=1e-05))
    

@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestExp(flow.unittest.TestCase):
    def test_exp(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_exp,
            _test_tensor_exp,
            _test_exp_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
