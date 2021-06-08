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


def l1_loss_1d(input, target, reduction="none"):
    assert len(input.shape) == target.shape
    assert (
        input.shape == target.shape), "The Input shape must be the same as Target shape"

    np_l1 = np.abs(target - input)
    
    if reduction == "mean":
        return np.mean(np_l1)
    elif reduction == "sum":
        return np.sum(np_l1)
    else:
        return np_l1

def _test_l1loss(test_case, device):
    x = np.array([[1, 1, 1], [2, 2, 2], [7, 7, 7]]).astype(np.float32)
    y = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]]).astype(np.float32)
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    l1_loss = flow.nn.L1Loss(reduction="none")
    l1_loss = l1_loss.to(device)
    of_out = l1_loss(input, target)
    np_out = l1_loss_1d(input.numpy(), target.numpy(), reduction="none")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


def _test_l1loss_mean(test_case, device):
    x = np.array([[1, 1, 1], [2, 2, 2], [7, 7, 7]]).astype(np.float32)
    y = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]]).astype(np.float32)
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    l1_loss = flow.nn.L1Loss(reduction="mean")
    l1_loss = l1_loss.to(device)
    of_out = l1_loss(input, target)
    np_out = l1_loss_1d(input.numpy(), target.numpy(), reduction="mean")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

def _test_l1loss_sum(test_case, device):
    x = np.array([[1, 1, 1], [2, 2, 2], [7, 7, 7]]).astype(np.float32)
    y = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]]).astype(np.float32)
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    l1_loss = flow.nn.L1Loss(reduction="sum")
    l1_loss = l1_loss.to(device)
    of_out = l1_loss(input, target)
    np_out = l1_loss_1d(input.numpy(), target.numpy(), reduction="sum")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestL1LossModule(flow.unittest.TestCase):
    def test_nllloss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_l1loss,
            _test_l1loss_mean,
            _test_l1loss_sum
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
    