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
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="none")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_mean_backward(test_case, device):
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
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="mean")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [-0.20000000298023224, 0.0, 0.0],
        [0.0, 0.0, -0.20000000298023224],
        [0.0, -0.20000000298023224, 0.0],
        [0.0, -0.20000000298023224, 0.0],
        [-0.20000000298023224, 0.0, 0.0],
    ]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_sum_backward(test_case, device):
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
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="sum")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_segmentation_none_backward(test_case, device):
    x = np.array(
        [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
    ).astype(np.float32)
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="none")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[0.0, -1.0], [-1.0, 0.0]], [[-1.0, 0.0], [0.0, -1.0]]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_segmentation_mean_backward(test_case, device):
    x = np.array(
        [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
    ).astype(np.float32)
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="mean")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[0.0, -0.25], [-0.25, 0.0]], [[-0.25, 0.0], [0.0, -0.25]]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_segmentation_sum_backward(test_case, device):
    x = np.array(
        [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
    ).astype(np.float32)
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="sum")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[0.0, -1.0], [-1.0, 0.0]], [[-1.0, 0.0], [0.0, -1.0]]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_bert_none_backward(test_case, device):
    x = np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]).astype(
        np.float32
    )
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[1, 0, 0, 1]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="none")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[0.0, -1.0, -1.0, 0.0], [-1.0, 0.0, 0.0, -1.0]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_bert_mean_backward(test_case, device):
    x = np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]).astype(
        np.float32
    )
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[1, 0, 0, 1]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="mean")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[0.0, -0.25, -0.25, 0.0], [-0.25, 0.0, 0.0, -0.25]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_bert_sum_backward(test_case, device):
    x = np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]).astype(
        np.float32
    )
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[1, 0, 0, 1]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="sum")
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[0.0, -1.0, -1.0, 0.0], [-1.0, 0.0, 0.0, -1.0]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_none_backward_with_ignore_index(test_case, device):
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
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="none", ignore_index=1)
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_mean_backward_with_ignore_index(test_case, device):
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
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="mean", ignore_index=1)
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [-0.33333, 0.0, 0.0],
        [0.0, 0.0, -0.33333],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.33333, 0.0, 0.0],
    ]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_sum_backward_with_ignore_index(test_case, device):
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
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="sum", ignore_index=1)
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_segmentation_none_backward_with_ignore_index(test_case, device):
    x = np.array(
        [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
    ).astype(np.float32)
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="none", ignore_index=1)
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[0.0, -1.0], [-1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_segmentation_mean_backward_with_ignore_index(test_case, device):
    x = np.array(
        [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
    ).astype(np.float32)
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="mean", ignore_index=1)
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[0.0, -0.5], [-0.5, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_segmentation_sum_backward_with_ignore_index(test_case, device):
    x = np.array(
        [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
    ).astype(np.float32)
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="sum", ignore_index=1)
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[0.0, -1.0], [-1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_bert_none_backward_with_ignore_index(test_case, device):
    x = np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]).astype(
        np.float32
    )
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[1, 0, 0, 1]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="none", ignore_index=1)
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[0.0, -1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_bert_mean_backward_with_ignore_index(test_case, device):
    x = np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]).astype(
        np.float32
    )
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[1, 0, 0, 1]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="mean", ignore_index=1)
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[0.0, -0.5, -0.5, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_nllloss_bert_sum_backward_with_ignore_index(test_case, device):
    x = np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]).astype(
        np.float32
    )
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = np.array([[1, 0, 0, 1]]).astype(np.int)
    target = flow.Tensor(y, dtype=flow.int64, device=flow.device(device))
    nll_loss = flow.nn.NLLLoss(reduction="sum", ignore_index=1)
    nll_loss = nll_loss.to(device)
    of_out = nll_loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[0.0, -1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


@flow.unittest.skip_unless_1n1d()
class TestNLLLossModule(flow.unittest.TestCase):
    def test_nllloss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_nllloss_none_backward,
            _test_nllloss_mean_backward,
            _test_nllloss_sum_backward,
            _test_nllloss_segmentation_none_backward,
            _test_nllloss_segmentation_mean_backward,
            _test_nllloss_segmentation_sum_backward,
            _test_nllloss_bert_none_backward,
            _test_nllloss_bert_mean_backward,
            _test_nllloss_bert_sum_backward,
            _test_nllloss_none_backward_with_ignore_index,
            _test_nllloss_mean_backward_with_ignore_index,
            _test_nllloss_sum_backward_with_ignore_index,
            _test_nllloss_segmentation_none_backward_with_ignore_index,
            _test_nllloss_segmentation_mean_backward_with_ignore_index,
            _test_nllloss_segmentation_sum_backward_with_ignore_index,
            _test_nllloss_bert_none_backward_with_ignore_index,
            _test_nllloss_bert_mean_backward_with_ignore_index,
            _test_nllloss_bert_sum_backward_with_ignore_index,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
