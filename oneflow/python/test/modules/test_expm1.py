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

import oneflow.experimental as flow
from test_util import GenArgList


def _test_expm1_forward(test_case, shape, device):
    x = flow.Tensor(np.random.randn(*shape), device=flow.device(device))

    of_out = flow.expm1(x)
    np_out = np.expm1(x.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


    x = flow.Tensor(np.random.randn(*shape), requires_grad=True, device=flow.device(device))
    of_out = x.expm1()
    np_out = np.expm1(x.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_expm1_backward(test_case, shape, device):
    x = flow.Tensor(np.random.randn(*shape), requires_grad=True, device=flow.device(device))

    of_out = flow.expm1(x)
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.exp(x.numpy()), 1e-4, 1e-4))


    x = flow.Tensor(np.random.randn(*shape), requires_grad=True, device=flow.device(device))
    of_out = x.expm1()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.exp(x.numpy()), 1e-4, 1e-4))



@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAddModule(flow.unittest.TestCase):
    def test_expm1(test_case):
        arg_dict_forward = OrderedDict()
        arg_dict_forward["shape"] = [(2, ), (2, 3), (2, 4, 5, 6)]
        arg_dict_forward["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict_forward):
            _test_expm1_forward(test_case, *arg)

        arg_dict_backward = OrderedDict()
        arg_dict_backward["shape"] = [(1,)]
        arg_dict_backward["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict_backward):
            _test_expm1_backward(test_case, *arg)
        



if __name__ == "__main__":
    unittest.main()
    