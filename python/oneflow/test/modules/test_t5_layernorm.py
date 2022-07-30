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


class TorchT5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        res = self.weight * hidden_states
        return res


def _test_t5_layer_norm(test_case, device):
    torch_t5_layernrom = TorchT5LayerNorm(3)
    oneflow_t5_layernorm = flow.nn.RMSLayerNorm(3)
    torch_t5_layernrom.to(device)
    oneflow_t5_layernorm.to(device)
    x = np.random.randn(2, 4, 3)
    torch_x = torch.tensor(x, requires_grad=True, device=torch.device(device))
    oneflow_x = flow.tensor(x, requires_grad=True, device=flow.device(device))
    torch_y = torch_t5_layernrom(torch_x)
    oneflow_y = oneflow_t5_layernorm(oneflow_x)
    test_case.assertTrue(
        np.allclose(
            torch_y.detach().cpu().numpy(), oneflow_y.numpy(), rtol=1e-4, atol=1e-4
        )
    )
    torch_y_sum = torch_y.sum()
    torch_y_sum.backward()
    oneflow_y_sum = oneflow_y.sum()
    oneflow_y_sum.backward()
    test_case.assertTrue(
        np.allclose(
            torch_x.grad.cpu().numpy(), oneflow_x.grad.numpy(), rtol=1e-5, atol=1e-5
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_t5_layernorm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_t5_layer_norm]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
