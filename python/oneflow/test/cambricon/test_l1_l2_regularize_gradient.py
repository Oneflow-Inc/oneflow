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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


def _test_l1_l2_regularize_gradient(test_case, shape, device, dtype, optimizer):
    model_np = np.random.randn(*shape)
    input_np = np.random.randn(*shape)
    lr = np.random.uniform()
    weight_decay = np.random.uniform()

    def _get_updated_param(device):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = flow.nn.Parameter(
                    flow.tensor(model_np, device=flow.device(device), dtype=dtype)
                )

            def forward(self, input):
                return self.weight * input

        model = TestModule().to(device)
        optim = optimizer(
            [model.weight], lr=lr, momentum=0.9, weight_decay=weight_decay
        )

        class TestGraph(nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = model
                self.add_optimizer(optim)

            def build(self, input):
                loss = self.m(input).sum()
                loss.backward()
                return loss

        input = flow.tensor(input_np, device=flow.device(device), dtype=dtype)
        loss = TestGraph()(input)
        return model.weight

    model_updated_cpu = _get_updated_param("cpu")
    model_updated_mlu = _get_updated_param("mlu")

    test_case.assertTrue(
        np.allclose(
            model_updated_cpu.numpy(), model_updated_mlu.numpy(), 0.0001, 0.0001
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestL1L2RegularizeGradientMLUModule(flow.unittest.TestCase):
    def test_l1_l2_regularize_gradient(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_l1_l2_regularize_gradient,
        ]
        arg_dict["shape"] = [(1,), (1, 2023), (200, 200), (400, 400)]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [flow.float32]
        arg_dict["optimizer"] = [flow.optim.SGD]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
