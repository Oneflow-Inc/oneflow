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


def _np_bcewithlogitsloss(
    np_input, np_target, np_weight=None, np_pos_weight=None, reduction="none"
):
    _neg_input = np.negative(np_input)
    _max_val = np.clip(_neg_input, 0, None)
    _neg_max_val = np.negative(_max_val)
    if np_pos_weight is not None:
        _log_weight = (np_pos_weight - 1) * np_target + 1
        _loss = (1 - np_target) * np_input + _log_weight * (
            np.log(np.exp(_neg_max_val) + np.exp(_neg_input - _max_val)) + _max_val
        )
    else:
        _loss = (1 - np_target) * np_input + _max_val
        _loss += np.log(np.exp(_neg_max_val) + np.exp(_neg_input - _max_val))
    if np_weight is not None:
        assert (
            np_weight.shape == np_input.shape
        ), "The weight shape must be the same as Input shape"
        _weighted_loss = np_weight * _loss
    else:
        _weighted_loss = _loss
    if reduction == "mean":
        return _weighted_loss.mean()
    elif reduction == "sum":
        return _weighted_loss.sum()
    else:
        return _weighted_loss


def _np_bcewithlogitsloss_grad(np_input, np_target, np_weight, np_pos_weight):
    elemcnt = np_target.size
    np_bce_with_logits_grad_mean = -(np_weight / elemcnt) * (
        np_target
        - 1
        + ((1 - np_pos_weight) * np_target - 1)
        * (-np.exp(-np_input) / (1 + np.exp(-np_input)))
    )
    np_bce_with_logits_grad_sum = np_bce_with_logits_grad_mean * elemcnt
    return {
        "mean": np_bce_with_logits_grad_mean,
        "sum": np_bce_with_logits_grad_sum,
        "none": np_bce_with_logits_grad_sum,
    }


def _test_bcewithlogitsloss_impl(test_case, device, shape, reduction):
    x = np.random.randn(*shape).astype(np.float32)
    y = np.random.randint(0, 2, [*shape]).astype(np.float32)
    w = np.random.randn(*shape).astype(np.float32)
    pw = np.random.randn([*shape][-1]).astype(np.float32)
    input = flow.Tensor(
        x, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    weight = flow.Tensor(w, dtype=flow.float32, device=flow.device(device))
    pos_weight = flow.Tensor(pw, dtype=flow.float32, device=flow.device(device))
    bcewithlogits_loss = flow.nn.BCEWithLogitsLoss(
        weight=weight, pos_weight=pos_weight, reduction=reduction
    )
    of_out = bcewithlogits_loss(input, target)
    np_out = _np_bcewithlogitsloss(
        x, y, np_weight=w, np_pos_weight=pw, reduction=reduction
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = _np_bcewithlogitsloss_grad(x, y, np_weight=w, np_pos_weight=pw)[reduction]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestBCEWithLogitsLossModule(flow.unittest.TestCase):
    def test_bcewithlogitsloss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_bcewithlogitsloss_impl]
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
