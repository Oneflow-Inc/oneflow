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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _np_bceloss(np_input, np_target, np_weight):
    np_cross_entropy = -(
        np_target * np.log(np_input) + (1 - np_target) * np.log(1 - np_input)
    )
    if np_weight is not None:
        assert (
            np_weight.shape == np_input.shape
        ), "The weight shape must be the same as Input shape"
        np_weighted_loss = np_weight * np_cross_entropy
    else:
        np_weighted_loss = np_cross_entropy
    np_bce_loss = np_weighted_loss
    np_bce_loss_sum = np.sum(np_weighted_loss)
    np_bce_loss_mean = np.mean(np_weighted_loss)
    return {"none": np_bce_loss, "sum": np_bce_loss_sum, "mean": np_bce_loss_mean}


def _test_bceloss_impl(test_case, device, reduction):
    x = np.array([[1.2, 0.2, -0.3], [0.7, 0.6, -2]]).astype(np.float32)
    y = np.array([[0, 1, 0], [1, 0, 1]]).astype(np.float32)
    w = np.array([[2, 2, 2], [2, 2, 2]]).astype(np.float32)
    input = flow.Tensor(
        x, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    weight = flow.Tensor(w, dtype=flow.float32, device=flow.device(device))
    activation = flow.nn.Sigmoid()
    sigmoid_input = activation(input)
    loss = flow.nn.BCELoss(weight, reduction=reduction)
    loss = loss.to(device)
    of_out = loss(sigmoid_input, target)
    np_out = _np_bceloss(sigmoid_input.numpy(), y, w)[reduction]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    if reduction == "none":
        np_grad = np.array(
            [[1.5370497, -0.90033215, 0.851115], [-0.6636245, 1.2913125, -1.7615942]]
        ).astype(np.float32)
    elif reduction == "sum":
        np_grad = np.array(
            [[1.5370497, -0.90033215, 0.851115], [-0.6636245, 1.2913125, -1.7615942]]
        ).astype(np.float32)
    else:
        np_grad = np.array(
            [
                [0.25617492, -0.15005533, 0.14185251],
                [-0.11060409, 0.21521877, -0.29359904],
            ]
        ).astype(np.float32)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))
    input_none = input = flow.Tensor(
        np.array([[1.2, 0.2, -0.3], [0.7, 0.6, -2]]).astype(np.float32),
        dtype=flow.float32,
        requires_grad=True,
        device=flow.device(device),
    )
    sigmoid_input_none = activation(input_none)
    loss_none = flow.nn.BCELoss(reduction=reduction)
    loss_none = loss_none.to(device)
    of_out_none = loss_none(sigmoid_input_none, target)
    np_out_none = _np_bceloss(sigmoid_input.numpy(), y, None)[reduction]
    test_case.assertTrue(np.allclose(of_out_none.numpy(), np_out_none, 1e-05, 1e-05))
    of_out_none = of_out_none.sum()
    of_out_none.backward()
    if reduction == "none":
        np_grad_none = np.array(
            [[0.7685, -0.4502, 0.4256], [-0.3318, 0.6457, -0.8808]]
        ).astype(np.float32)
    elif reduction == "sum":
        np_grad_none = np.array(
            [[0.7685, -0.4502, 0.4256], [-0.3318, 0.6457, -0.8808]]
        ).astype(np.float32)
    else:
        np_grad_none = np.array(
            [[0.1281, -0.075, 0.0709], [-0.0553, 0.1076, -0.1468]]
        ).astype(np.float32)
    test_case.assertTrue(
        np.allclose(input_none.grad.numpy(), np_grad_none, 0.0001, 0.0001)
    )
    input_none = input = flow.Tensor(
        np.array([[1.2, 0.2, -0.3], [0.7, 0.6, -2]]).astype(np.float32),
        dtype=flow.float32,
        requires_grad=True,
        device=flow.device(device),
    )
    sigmoid_input_none = activation(input_none)
    loss_none = flow.nn.BCELoss(reduction=reduction)
    loss_none = loss_none.to(device)
    of_out_none = loss_none(sigmoid_input_none, target)
    np_out_none = _np_bceloss(sigmoid_input.numpy(), y, None)[reduction]
    test_case.assertTrue(np.allclose(of_out_none.numpy(), np_out_none, 1e-05, 1e-05))
    of_out_none = of_out_none.sum()
    of_out_none.backward()
    if reduction == "none":
        np_grad_none = np.array(
            [[0.7685, -0.4502, 0.4256], [-0.3318, 0.6457, -0.8808]]
        ).astype(np.float32)
    elif reduction == "sum":
        np_grad_none = np.array(
            [[0.7685, -0.4502, 0.4256], [-0.3318, 0.6457, -0.8808]]
        ).astype(np.float32)
    else:
        np_grad_none = np.array(
            [[0.1281, -0.075, 0.0709], [-0.0553, 0.1076, -0.1468]]
        ).astype(np.float32)
    test_case.assertTrue(
        np.allclose(input_none.grad.numpy(), np_grad_none, 0.0001, 0.0001)
    )


@flow.unittest.skip_unless_1n1d()
class TestBCELossModule(flow.unittest.TestCase):
    def test_bceloss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_bceloss_impl]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["reduction"] = ["none", "sum", "mean"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
