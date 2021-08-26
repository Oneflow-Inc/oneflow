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


def np_margin_ranking_loss(margin, input1, input2, targets, reduction="none"):
    out = np.clip(margin + -targets * (input1 - input2), a_min=0, a_max=None)
    if reduction == "sum":
        return np.sum(out)
    elif reduction == "mean":
        return out.mean()
    elif reduction == "none":
        return out


def np_margin_ranking_loss_grad(margin, input1, input2, targets):
    out = np.clip(margin + -targets * (input1 - input2), a_min=0, a_max=None)
    out_grad1 = -1 * np.zeros_like(targets)
    out_grad2 = np.zeros_like(targets)
    out_grad1[np.nonzero(out)] = -targets[np.nonzero(out)]
    out_grad2[np.nonzero(out)] = targets[np.nonzero(out)]
    return (out_grad1, out_grad2)


def _test_marginrankingloss_none(test_case, shape, margin, device):
    input1 = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    target_pos = flow.Tensor(
        np.ones(shape), dtype=flow.float32, device=flow.device(device)
    )
    target_neg = flow.Tensor(
        -1 * np.ones(shape), dtype=flow.float32, device=flow.device(device)
    )
    margin_ranking_loss = flow.nn.MarginRankingLoss(margin=margin, reduction="none")
    margin_ranking_loss = margin_ranking_loss.to(device)
    of_out_pos = margin_ranking_loss(input1, input2, target_pos)
    np_out_pos = np_margin_ranking_loss(
        margin, input1.numpy(), input2.numpy(), target_pos.numpy(), reduction="none"
    )
    test_case.assertTrue(np.allclose(of_out_pos.numpy(), np_out_pos, 1e-05, 1e-05))
    of_out_neg = margin_ranking_loss(input1, input2, target_neg)
    np_out_neg = np_margin_ranking_loss(
        margin, input1.numpy(), input2.numpy(), target_neg.numpy(), reduction="none"
    )
    test_case.assertTrue(np.allclose(of_out_neg.numpy(), np_out_neg, 1e-05, 1e-05))


def _test_marginrankingloss_mean(test_case, shape, margin, device):
    input1 = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    target_pos = flow.Tensor(
        np.ones(shape), dtype=flow.float32, device=flow.device(device)
    )
    target_neg = flow.Tensor(
        -1 * np.ones(shape), dtype=flow.float32, device=flow.device(device)
    )
    margin_ranking_loss = flow.nn.MarginRankingLoss(margin=margin, reduction="mean")
    margin_ranking_loss = margin_ranking_loss.to(device)
    of_out_pos = margin_ranking_loss(input1, input2, target_pos)
    np_out_pos = np_margin_ranking_loss(
        margin, input1.numpy(), input2.numpy(), target_pos.numpy(), reduction="mean"
    )
    test_case.assertTrue(np.allclose(of_out_pos.numpy(), np_out_pos, 1e-05, 1e-05))
    of_out_neg = margin_ranking_loss(input1, input2, target_neg)
    np_out_neg = np_margin_ranking_loss(
        margin, input1.numpy(), input2.numpy(), target_neg.numpy(), reduction="mean"
    )
    test_case.assertTrue(np.allclose(of_out_neg.numpy(), np_out_neg, 1e-05, 1e-05))


def _test_marginrankingloss_sum(test_case, shape, margin, device):
    input1 = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    target_pos = flow.Tensor(
        np.ones(shape), dtype=flow.float32, device=flow.device(device)
    )
    target_neg = flow.Tensor(
        -1 * np.ones(shape), dtype=flow.float32, device=flow.device(device)
    )
    margin_ranking_loss = flow.nn.MarginRankingLoss(margin=margin, reduction="sum")
    margin_ranking_loss = margin_ranking_loss.to(device)
    of_out_pos = margin_ranking_loss(input1, input2, target_pos)
    np_out_pos = np_margin_ranking_loss(
        margin, input1.numpy(), input2.numpy(), target_pos.numpy(), reduction="sum"
    )
    test_case.assertTrue(np.allclose(of_out_pos.numpy(), np_out_pos, 1e-05, 1e-05))
    of_out_neg = margin_ranking_loss(input1, input2, target_neg)
    np_out_neg = np_margin_ranking_loss(
        margin, input1.numpy(), input2.numpy(), target_neg.numpy(), reduction="sum"
    )
    test_case.assertTrue(np.allclose(of_out_neg.numpy(), np_out_neg, 1e-05, 1e-05))


def _test_marginrankingloss_grad(test_case, shape, margin, device):
    input1 = flow.Tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input2 = flow.Tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    target = flow.Tensor(np.ones(shape), dtype=flow.float32, device=flow.device(device))
    margin_ranking_loss = flow.nn.MarginRankingLoss(margin=margin, reduction="sum")
    margin_ranking_loss = margin_ranking_loss.to(device)
    of_out = margin_ranking_loss(input1, input2, target)
    of_out.backward()
    (np_out_grad1, np_out_grad2) = np_margin_ranking_loss_grad(
        margin, input1.numpy(), input2.numpy(), target.numpy()
    )
    test_case.assertTrue(np.allclose(input1.grad.numpy(), np_out_grad1, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(input2.grad.numpy(), np_out_grad2, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestMarginRankingLossModule(flow.unittest.TestCase):
    def test_margin_ranking_loss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_marginrankingloss_none,
            _test_marginrankingloss_mean,
            _test_marginrankingloss_sum,
            _test_marginrankingloss_grad,
        ]
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["margin"] = [1.0, 0.3, 10]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
