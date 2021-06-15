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

import torch


def _np_bceloss(np_input, np_target, np_weight):
    np_cross_entropy = -(np_target * np.log(np_input) + (1 - np_target) * np.log(1 - np_input))
    
    if np_weight is not None:
            assert (
                np_weight.shape == np_input.shape
            ), "The weight shape must be the same as Input shape"
            np_weighted_loss = np_weight * np_cross_entropy
    else:
        np_weighted_loss = np_cross_entropy

    return {
        "none": np_weighted_loss,
        "mean": np.mean(np_weighted_loss),
        "sum": np.sum(np_weighted_loss),
    }


def _np_bce_grad(np_input, np_target, np_weight):
    np_cross_entropy_grad = -(
             (np_target - np_input)
            / ((1 - np_input) * np_input)
        )
    
    if np_weight is not None:
        np_weighted_grad = np_weight * np_cross_entropy_grad
    else:
        np_weighted_grad = np_cross_entropy_grad

    elem_cnt = np_input.size
    np_bce_grad_mean = np_weighted_grad / elem_cnt

    return {
        "none": np_weighted_grad,
        "mean": np_bce_grad_mean,
        "sum": np_weighted_grad,
    }

def _test_bceloss_impl(test_case, device, shape, reduction):
    x = np.array([[1.2, 0.2, -0.3], [0.7, 0.6, -2]]).astype(np.float32)
    y = np.array([[0, 1, 0], [1, 0, 1]]).astype(np.float32)
    w = np.array([[2, 2, 2], [2, 2, 2]]).astype(np.float32)

    input = flow.Tensor(x, dtype=flow.float32, requires_grad=True, device=flow.device(device))
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    weight = flow.Tensor(w, dtype=flow.float32, device=flow.device(device))

    activation = flow.nn.Sigmoid()
    sigmoid_input = activation(input)

    loss = flow.nn.BCELoss(reduction)
    loss = loss.to(device)
    of_out = loss(sigmoid_input, target, weight)

    np_out = _np_bceloss(sigmoid_input.numpy(), y, w)[reduction]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()
    np_grad = _np_bce_grad(sigmoid_input.numpy(), y, w)[reduction]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-3, 1e-3))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestBCELossModule(flow.unittest.TestCase):
    def test_bceloss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_bceloss_impl,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [
            (3, 5),
            (10, 9, 21),
            (14, 22, 9, 21),
            (3, 2, 4, 16, 5),
            (1,),
        ]
        arg_dict["reduction"] = [ "none", "sum", "mean"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
