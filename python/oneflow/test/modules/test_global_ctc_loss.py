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
import oneflow as flow
import oneflow.unittest
import torch
from oneflow.test_utils.automated_test_util.generators import *
from oneflow.test_utils.automated_test_util.torch_flow_dual_object import globaltest
from oneflow.test_utils.test_util import GenArgDict


def log_softmax(logits, axis=0):
    max_value = np.max(logits, axis, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis, keepdims=True)
    dist = exp / exp_sum
    return np.log(dist)


def _compare_torch_and_oneflow(
    test_case,
    torch_ctc_loss,
    flow_ctc_loss,
    placement,
    module_sbp,
    in_sbp,
    max_input_length,
    batch_size,
    num_classes,
    max_target_length,
):
    log_probs = np.random.random(
        size=(max_input_length, batch_size, num_classes)
    ).astype(np.float32)
    log_probs = log_softmax(log_probs, axis=2)
    targets = np.random.randint(
        1, high=num_classes, size=(batch_size, max_target_length), dtype=np.int32
    )
    input_lengths = np.random.randint(
        max_input_length / 2, high=max_input_length, size=(batch_size,), dtype=np.int32
    )
    target_lengths = np.random.randint(
        max_target_length / 2,
        high=max_target_length,
        size=(batch_size,),
        dtype=np.int32,
    )

    log_probs_torch = torch.tensor(log_probs, dtype=torch.float32, requires_grad=True)
    targets_torch = torch.tensor(targets, dtype=torch.int32)
    input_lengths_torch = torch.tensor(input_lengths, dtype=torch.int32)
    target_lengths_torch = torch.tensor(target_lengths, dtype=torch.int32)

    log_probs_flow = (
        flow.tensor(log_probs, dtype=flow.float32, requires_grad=True)
        .to_global(flow.placement.all("cpu"), flow.sbp.broadcast)
        .to_global(placement=placement, sbp=in_sbp)
    )
    targets_flow = (
        flow.tensor(targets, dtype=flow.int32)
        .to_global(flow.placement.all("cpu"), flow.sbp.broadcast)
        .to_global(placement=placement, sbp=in_sbp)
    )
    input_lengths_flow = (
        flow.tensor(input_lengths, dtype=flow.int32)
        .to_global(flow.placement.all("cpu"), flow.sbp.broadcast)
        .to_global(placement=placement, sbp=in_sbp)
    )
    target_lengths_flow = (
        flow.tensor(target_lengths, dtype=flow.int32)
        .to_global(flow.placement.all("cpu"), flow.sbp.broadcast)
        .to_global(placement=placement, sbp=in_sbp)
    )

    out_torch = torch_ctc_loss(
        log_probs_torch, targets_torch, input_lengths_torch, target_lengths_torch
    )
    out_flow = flow_ctc_loss(
        log_probs_flow, targets_flow, input_lengths_flow, target_lengths_flow
    )

    # check forward
    local_output = out_flow.to_global(
        placement=placement,
        sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))],
    ).to_local()
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.allclose(
                out_torch.cpu().detach().numpy(),
                local_output.numpy(),
                rtol=1e-05,
                atol=1e-05,
            )
        )

    # check backward
    out_torch.sum().backward()
    out_flow.sum().backward()
    local_x_grad = log_probs_flow.to_global(
        placement=placement,
        sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))],
    ).to_local()
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.allclose(
                log_probs_torch.cpu().detach().numpy(),
                local_x_grad.numpy(),
                rtol=1e-05,
                atol=1e-05,
            )
        )


def _test_ctc_loss_impl(
    test_case,
    placement,
    module_sbp,
    in_sbp,
    max_input_length,
    batch_size,
    num_classes,
    max_target_length,
    blank,
    reduction,
    zero_infinity,
):
    torch_ctc_loss = torch.nn.CTCLoss(
        blank=blank, reduction=reduction, zero_infinity=zero_infinity
    )
    flow_ctc_loss = flow.nn.CTCLoss(
        blank=blank, reduction=reduction, zero_infinity=zero_infinity
    )
    _compare_torch_and_oneflow(
        test_case,
        torch_ctc_loss,
        flow_ctc_loss,
        placement,
        module_sbp,
        in_sbp,
        max_input_length,
        batch_size,
        num_classes,
        max_target_length,
    )


@flow.unittest.skip_unless_1n2d()
class TestCTCLossGlobal(oneflow.unittest.TestCase):
    @globaltest
    def test_ctc_loss_global(test_case):
        arg_dict = OrderedDict()
        arg_dict["max_input_length"] = [20]
        arg_dict["batch_size"] = [4]
        arg_dict["num_classes"] = [5]
        arg_dict["max_target_length"] = [10]
        arg_dict["blank"] = [0, 4]
        arg_dict["reduction"] = ["mean", "none"]
        arg_dict["zero_infinity"] = [False, True]

        module_sbp = flow.sbp.broadcast
        for args in GenArgDict(arg_dict):
            for placement in all_placement():
                for in_sbp in all_sbp(placement):
                    _test_ctc_loss_impl(
                        test_case, placement, module_sbp, in_sbp, **args
                    )


if __name__ == "__main__":
    unittest.main()
