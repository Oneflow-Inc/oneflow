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

import oneflow as flow
import oneflow.unittest
from collections import OrderedDict
from test_util import GenArgList


shapes = {2: (128, 8), 3: (16, 8, 64), 4: (16, 8, 32, 32), 5: (16, 8, 16, 16, 16)}


def compare_loss(device_type, dim, reduction, cls, data_generator):
    x, y = data_generator(dim, device_type)
    f = cls(reduction=reduction).to(device_type)
    z_eager = f(x, y)

    class CurrentGraph(flow.nn.Graph):
        def __init__(self) -> None:
            super().__init__()
            self.f = f

        def build(self, x, y):
            return self.f(x, y)

    f_g = CurrentGraph()
    z_lazy = f_g(x, y)
    assert np.allclose(z_eager.numpy(), z_lazy.numpy(), rtol=1.0e-5, atol=1.0e-5)


def generate_necessity_default(dim: int, device: str):
    shape = shapes[dim]
    x_np = np.random.uniform(0, 1, shape)
    y_np = np.random.uniform(0, 1, shape)
    x = flow.tensor(x_np, dtype=flow.float32, device=device)
    y = flow.tensor(y_np, dtype=flow.float32, device=device)
    return x, y


def generate_necessity_for_cross_entropy_or_nll_loss(dim: int, device: str):
    shape = shapes[dim]
    y_shape = (shape[0],) if dim == 2 else (shape[0], *shape[2:])
    x_np = np.random.uniform(0, 1, shape)
    y_np = np.random.randint(0, shape[1], y_shape)
    x = flow.tensor(x_np, dtype=flow.float32, device=device)
    y = flow.tensor(y_np, dtype=flow.int32, device=device)
    return x, y


@flow.unittest.skip_unless_1n1d()
class TestKLDivLossGraph(oneflow.unittest.TestCase):
    def test_kl_div_loss_graph(testcase):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["dim"] = [2, 3, 4, 5]
        arg_dict["reduction"] = ["sum", "mean"]
        arg_dict["cls"] = [flow.nn.KLDivLoss]
        arg_dict["data_generator"] = [generate_necessity_default]
        for arg in GenArgList(arg_dict):
            compare_loss(*arg)


@flow.unittest.skip_unless_1n1d()
class TestSmoothL1LossGraph(oneflow.unittest.TestCase):
    def test_smooth_l1_loss_graph(testcase):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["dim"] = [2, 3, 4, 5]
        arg_dict["reduction"] = ["sum", "mean"]
        arg_dict["cls"] = [flow.nn.SmoothL1Loss]
        arg_dict["data_generator"] = [generate_necessity_default]
        for arg in GenArgList(arg_dict):
            compare_loss(*arg)


@flow.unittest.skip_unless_1n1d()
class TestBCELossOrWithLogitsGraph(flow.unittest.TestCase):
    def test_bce_loss_graph(testcase):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["dim"] = [2, 3, 4, 5]
        arg_dict["reduction"] = ["sum", "mean"]
        arg_dict["cls"] = [flow.nn.BCELoss, flow.nn.BCEWithLogitsLoss]
        arg_dict["data_generator"] = [generate_necessity_default]
        for arg in GenArgList(arg_dict):
            compare_loss(*arg)


@flow.unittest.skip_unless_1n1d()
class TestCrossEntropyOrNllLossGraph(flow.unittest.TestCase):
    def test_cross_entropy_loss_or_nll_loss_graph(testcase):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["dim"] = [2, 3, 4, 5]
        arg_dict["reduction"] = ["sum", "mean"]
        arg_dict["cls"] = [flow.nn.CrossEntropyLoss, flow.nn.NLLLoss]
        arg_dict["data_generator"] = [generate_necessity_for_cross_entropy_or_nll_loss]
        for arg in GenArgList(arg_dict):
            compare_loss(*arg)


if __name__ == "__main__":
    unittest.main()
