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
from oneflow.test_utils.test_util import GenArgList


def get_sbp(device: str):
    return flow.placement.all(device), flow.sbp.split(0)


shapes = {2: (128, 8), 3: (16, 8, 64), 4: (16, 8, 32, 32), 5: (16, 8, 16, 16, 16)}


def compare_loss(device_type, dim, reduction, cls, data_generator):
    x, y, x1, y1 = data_generator(dim, device_type, *get_sbp(device_type))
    reduce_loss_func = cls(reduction=reduction).to(device_type)
    none_loss_func = cls(reduction="none").to(device_type)

    loss_mean = reduce_loss_func(x, y)
    loss_none = (
        flow.mean(none_loss_func(x1, y1))
        if reduction == "mean"
        else flow.sum(none_loss_func(x1, y1))
    )

    loss_mean.backward()
    loss_none.backward()

    assert np.allclose(
        loss_none.to_local().numpy(),
        loss_mean.to_local().numpy(),
        rtol=1e-05,
        atol=1e-05,
    )
    assert np.allclose(loss_none.numpy(), loss_mean.numpy(), rtol=1e-05, atol=1e-05,)
    assert np.allclose(
        x.grad.to_local().numpy(), x1.grad.to_local().numpy(), rtol=1e-05, atol=1e-05,
    )


def generate_necessity_default(dim: int, device: str, placement, sbp):
    shape = shapes[dim]
    x_np = np.random.uniform(0, 1, shape)
    y_np = np.random.uniform(0, 1, shape)

    def f(x, requires_grad):
        t = flow.tensor(x, device=device, requires_grad=requires_grad).to_global(
            placement=placement, sbp=[sbp]
        )
        if requires_grad:
            t.retain_grad()
        return t

    return f(x_np, True), f(y_np, False), f(x_np, True), f(y_np, False)


def generate_necessity_for_cross_entropy_or_nll_loss(
    dim: int, device: str, placement, sbp
):
    shape = shapes[dim]
    y_shape = (shape[0],) if dim == 2 else (shape[0], *shape[2:])
    x_np = np.random.uniform(0, 1, shape)
    y_np = np.random.randint(0, shape[1], y_shape)

    def f(x, requires_grad):
        t = flow.tensor(x, device=device, requires_grad=requires_grad).to_global(
            placement=placement, sbp=[sbp]
        )
        if requires_grad:
            t.retain_grad()
        return t

    return f(x_np, True), f(y_np, False), f(x_np, True), f(y_np, False)


class TestBCELossOrWithLogitsConsistent(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_bce_loss(testcase):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["dim"] = [2, 3, 4, 5]
        arg_dict["reduction"] = ["sum", "mean"]
        arg_dict["cls"] = [flow.nn.BCELoss, flow.nn.BCEWithLogitsLoss]
        arg_dict["data_generator"] = [generate_necessity_default]
        for arg in GenArgList(arg_dict):
            compare_loss(*arg)


class TestCrossEntropyOrNllLossConsistent(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_cross_entropy_loss_or_nll_loss(testcase):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["dim"] = [2, 3, 4, 5]
        arg_dict["reduction"] = ["sum", "mean"]
        arg_dict["cls"] = [flow.nn.CrossEntropyLoss, flow.nn.NLLLoss]
        arg_dict["data_generator"] = [generate_necessity_for_cross_entropy_or_nll_loss]
        for arg in GenArgList(arg_dict):
            compare_loss(*arg)


class TestKLDivLossConsistent(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_kl_div_loss(testcase):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["dim"] = [2, 3, 4, 5]
        arg_dict["reduction"] = ["sum", "mean"]
        arg_dict["cls"] = [flow.nn.KLDivLoss]
        arg_dict["data_generator"] = [generate_necessity_default]
        for arg in GenArgList(arg_dict):
            compare_loss(*arg)


class TestSmoothL1LossConsistent(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_smooth_l1_loss(testcase):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["dim"] = [2, 3, 4, 5]
        arg_dict["reduction"] = ["sum", "mean"]
        arg_dict["cls"] = [flow.nn.SmoothL1Loss]
        arg_dict["data_generator"] = [generate_necessity_default]
        for arg in GenArgList(arg_dict):
            compare_loss(*arg)


if __name__ == "__main__":
    unittest.main()
