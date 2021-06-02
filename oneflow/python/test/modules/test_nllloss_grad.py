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


def _test_nllloss_none_backward(test_case, device):
    x = np.array(
        [
            [0.88103855, 0.9908683, 0.6226845],
            [0.53331435, 0.07999352, 0.8549948],
            [0.25879037, 0.39530203, 0.698465],
            [0.73427284, 0.63575995, 0.18827209],
            [0.05689114, 0.0862954, 0.6325046],
        ]
    ).astype(np.float32)
    y = np.array([0, 2, 1, 1, 0]).astype(np.int)
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )

    target = flow.Tensor(
        y, dtype=flow.int64, device=flow.device(device), requires_grad=True
    )
    nll_loss = flow.nn.NLLLoss()
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    print(input.grad.numpy())


def _test_nllloss_mean(test_case, device):
    x = np.array(
        [
            [0.88103855, 0.9908683, 0.6226845],
            [0.53331435, 0.07999352, 0.8549948],
            [0.25879037, 0.39530203, 0.698465],
            [0.73427284, 0.63575995, 0.18827209],
            [0.05689114, 0.0862954, 0.6325046],
        ]
    ).astype(np.float32)
    y = np.array([0, 2, 1, 1, 0]).astype(np.int)
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))

    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="mean")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    np_out = nll_loss_1d(input.numpy(), target.numpy(), reduction="mean")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


def _test_nllloss_sum(test_case, device):
    x = np.array(
        [
            [0.88103855, 0.9908683, 0.6226845],
            [0.53331435, 0.07999352, 0.8549948],
            [0.25879037, 0.39530203, 0.698465],
            [0.73427284, 0.63575995, 0.18827209],
            [0.05689114, 0.0862954, 0.6325046],
        ]
    ).astype(np.float32)
    y = np.array([0, 2, 1, 1, 0]).astype(np.int)
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))

    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="sum")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    np_out = nll_loss_1d(input.numpy(), target.numpy(), reduction="sum")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


def _test_nllloss_segmentation_none(test_case, device):
    x = np.array(
        [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
    ).astype(np.float32)
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss()
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    np_out = nll_loss_2d(input.numpy(), target.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


def _test_nllloss_segmentation_mean(test_case, device):
    x = np.array(
        [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
    ).astype(np.float32)
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="mean")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    np_out = nll_loss_2d(input.numpy(), target.numpy(), reduction="mean")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


def _test_nllloss_segmentation_sum(test_case, device):
    x = np.array(
        [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
    ).astype(np.float32)
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="sum")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    np_out = nll_loss_2d(input.numpy(), target.numpy(), reduction="sum")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


def _test_nllloss_bert_none(test_case, device):
    x = np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]).astype(
        np.float32
    )
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    y = np.array([[1, 0, 0, 1]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss()
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    np_out = nll_loss_bert(input.numpy(), target.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


def _test_nllloss_bert_mean(test_case, device):
    x = np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]).astype(
        np.float32
    )
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    y = np.array([[1, 0, 0, 1]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="mean")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    np_out = nll_loss_bert(input.numpy(), target.numpy(), reduction="mean")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


def _test_nllloss_bert_sum(test_case, device):
    x = np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]).astype(
        np.float32
    )
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    y = np.array([[1, 0, 0, 1]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="sum")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    np_out = nll_loss_bert(input.numpy(), target.numpy(), reduction="sum")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestNLLLossModule(flow.unittest.TestCase):
    def test_nllloss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_nllloss_none_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
