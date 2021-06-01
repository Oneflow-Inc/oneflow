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
from scipy import special as S

import oneflow.experimental as flow
from torch import per_tensor_affine
from test_util import GenArgList


def _test_lgamma(test_case, device):
    arr = np.array([0, 0.5, 1, 4.5, -4, -5.6])
    input = flow.Tensor(arr, dtype=flow.float32, requires_grad = True, device=flow.device(device))
    
    of_out = flow.lgamma(input)
    np_out = S.gammaln(arr)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()
    np_grad_out = S.digamma(arr)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad_out, 1e-5, 1e-5))

@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLess(flow.unittest.TestCase):
    def test_lgamma(test_case):
            arg_dict = OrderedDict()
            arg_dict["device"] = ["cpu", "cuda"]
            for arg in GenArgList(arg_dict):
                _test_lgamma(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
