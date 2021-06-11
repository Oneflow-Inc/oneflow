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


def _np_l1loss(np_input, np_target, np_weight, reduction):
    np_cross_entropy_loss = -(target * np.log(input) + (1 - target) * np.log(1 - input))
    
    if np_weight is not None:
            assert (
                np_weight.shape == np_input.shape
            ), "The weight shape must be the same as Input shape"
            np_weighted_loss = np_weight * np_cross_entropy_loss
        else:
            np_weighted_loss = np_cross_entropy_loss


    # return {
    #     "none": np_weighted_loss,
    #     "mean": np.mean(np_weighted_loss),
    #     "sum": np.sum(np_weighted_loss),
    # }

    if reduction == "mean":
        return np.mean(np_weighted_loss)
    elif reduction == "sum":
        return np.sum(np_weighted_loss
    else:
        return np_weighted_loss


def _np_l1loss_grad(np_input, np_target):
    elem_cnt = np_input.size
    np_grad = np.where(np_target - np_input > 0, -1, 1)
    np_l1_grad_sum = np_grad
    np_l1_grad_mean = np_l1_grad_sum / elem_cnt

    return {
        "none": np_grad,
        "mean": np_l1_grad_mean,
        "sum": np_l1_grad_sum,
    }
    

def _test_l1loss_impl(test_case, device, shape, reduction):
    x = np.random.randn(*shape)
    y = np.random.randn(*shape)
    input = flow.Tensor(
        x, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))

    loss = flow.nn.L1Loss(reduction)
    loss = loss.to(device)
    of_out = loss(input, target)
    np_out = _np_l1loss(x, y)[reduction]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()
    np_grad = _np_l1loss_grad(x, y)[reduction]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestL1LossModule(flow.unittest.TestCase):
    def test_l1loss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_l1loss_impl,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [
            (3, 5),
            (10, 9, 21),
            (14, 22, 9, 21),
            (3, 2, 4, 16, 5),
            (1,),
        ]
        arg_dict["reduction"] = ["none", "sum", "mean"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
