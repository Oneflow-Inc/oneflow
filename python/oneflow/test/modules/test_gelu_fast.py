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

import math
import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest
import torch


class FastGELUActivation(torch.nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


def _test_gelu_fast(test_case, device):
    torch_gelu_fast = FastGELUActivation()
    x = np.random.randn(2, 4, 3)
    torch_x = torch.tensor(x, requires_grad=True, device=torch.device(device))
    oneflow_x = flow.tensor(x, requires_grad=True, device=flow.device(device))
    torch_y = torch_gelu_fast(torch_x)
    oneflow_y = flow._C.gelu_fast(oneflow_x)
    test_case.assertTrue(np.allclose(torch_y.detach().cpu().numpy(), oneflow_y.numpy()))
    torch_y_sum = torch_y.sum()
    torch_y_sum.backward()
    oneflow_y_sum = oneflow_y.sum()
    oneflow_y_sum.backward()
    test_case.assertTrue(
        np.allclose(torch_x.grad.cpu().numpy(), oneflow_x.grad.numpy())
    )

@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_gelu_approximate(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_gelu_fast]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()