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

ninf = -float("inf")


def _logsumexp(a, b):
    if a < b:
        (a, b) = (b, a)
    if b == ninf:
        return a
    else:
        return a + np.log(1 + np.exp(b - a))


def logsumexp(*args):
    res = args[0]
    for e in args[1:]:
        res = _logsumexp(res, e)
    return res


def log_softmax(logits, axis=0):
    max_value = np.max(logits, axis, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis, keepdims=True)
    dist = exp / exp_sum
    return np.log(dist)


def get_target_prime(targets, b, s, blank):
    if s % 2 == 0:
        return blank
    else:
        return targets[b, s // 2]


def ctc_loss_np(log_probs, targets, input_lengths, target_lengths, blank=0):
    (max_input_length, batch_size, _) = log_probs.shape
    (_, max_target_length) = targets.shape
    loss = np.zeros(batch_size)
    alpha = np.zeros([batch_size, max_input_length, 2 * max_target_length + 1])
    alpha[:, 0] = ninf
    for b in range(0, batch_size):
        input_length = input_lengths[b]
        target_length = target_lengths[b]
        alpha[b, 0, 0] = log_probs[0, b, blank]
        if target_length > 0:
            current_target_prime = get_target_prime(targets, b, 1, blank)
            alpha[b, 0, 1] = log_probs[0, b, current_target_prime]
        for t in range(1, input_length):
            for s in range(0, 2 * target_length + 1):
                current_target_prime = get_target_prime(targets, b, s, blank)
                la1 = alpha[b, t - 1, s]
                if s > 0:
                    la2 = alpha[b, t - 1, s - 1]
                else:
                    la2 = ninf
                if (
                    s > 1
                    and get_target_prime(targets, b, s - 2, blank)
                    != current_target_prime
                ):
                    la3 = alpha[b, t - 1, s - 2]
                else:
                    la3 = ninf
                alpha[b, t, s] = (
                    logsumexp(la1, la2, la3) + log_probs[t, b, current_target_prime]
                )
        if target_length == 0:
            loss[b] = -alpha[b, input_length - 1, 0]
        else:
            l1 = alpha[b, input_length - 1, target_length * 2]
            l2 = alpha[b, input_length - 1, target_length * 2 - 1]
            loss[b] = -logsumexp(l1, l2)
    return (loss, alpha)


def ctc_loss_grad_np(
    grad_out,
    loss,
    alpha,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    zero_infinity=False,
):
    (max_input_length, batch_size, num_labels) = log_probs.shape
    (_, max_target_length) = targets.shape
    beta = np.zeros([batch_size, max_input_length, 2 * max_target_length + 1])
    grad = np.zeros(log_probs.shape, dtype=log_probs.dtype)
    grad.fill(ninf)
    for b in range(0, batch_size):
        input_length = input_lengths[b]
        target_length = target_lengths[b]
        nll = loss[b]
        if zero_infinity and nll == float("inf"):
            grad[:, b, :] = 0
            continue
        if input_length > 0:
            beta[b, input_length - 1, :] = ninf
            beta[b, input_length - 1, 2 * target_length] = log_probs[
                input_length - 1, b, blank
            ]
            grad[input_length - 1, b, blank] = (
                alpha[b, input_length - 1, 2 * target_length]
                + beta[b, input_length - 1, 2 * target_length]
            )
            if target_length > 0:
                current_target_prime = get_target_prime(
                    targets, b, 2 * target_length - 1, blank
                )
                beta[b, input_length - 1, 2 * target_length - 1] = log_probs[
                    input_length - 1, b, current_target_prime
                ]
                grad[input_length - 1, b, current_target_prime] = (
                    alpha[b, input_length - 1, 2 * target_length - 1]
                    + beta[b, input_length - 1, 2 * target_length - 1]
                )
        for t in range(input_length - 2, -1, -1):
            for s in range(2 * target_length, -1, -1):
                current_target_prime = get_target_prime(targets, b, s, blank)
                lb1 = beta[b, t + 1, s]
                if s < 2 * target_length:
                    lb2 = beta[b, t + 1, s + 1]
                else:
                    lb2 = ninf
                if (
                    s < 2 * target_length - 1
                    and get_target_prime(targets, b, s + 2, blank)
                    != current_target_prime
                ):
                    lb3 = beta[b, t + 1, s + 2]
                else:
                    lb3 = ninf
                beta[b, t, s] = (
                    logsumexp(lb1, lb2, lb3) + log_probs[t, b, current_target_prime]
                )
                alpha_beta = alpha[b, t, s] + beta[b, t, s]
                lcab = grad[t, b, current_target_prime]
                if lcab == ninf:
                    grad[t, b, current_target_prime] = alpha_beta
                else:
                    grad[t, b, current_target_prime] = logsumexp(lcab, alpha_beta)
        for t in range(0, input_length):
            for c in range(0, num_labels):
                res = grad[t, b, c]
                lp = log_probs[t, b, c]
                grad[t, b, c] = (np.exp(lp) - np.exp(res + nll - lp)) * grad_out[b]
        if input_length < max_input_length:
            grad[input_length:max_input_length, b] = 0
    return grad


def compare_with_np(
    device_type,
    device_num,
    data_type,
    max_input_length,
    batch_size,
    num_classes,
    max_target_length,
    blank,
    reduction,
    zero_infinity,
):
    assert data_type in ["float32", "double"]
    assert device_type in ["cuda", "cpu"]
    assert reduction in ["none", "mean", "sum"]
    assert zero_infinity in [False, True]
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
    (np_loss, np_alpha) = ctc_loss_np(
        log_probs, targets, input_lengths, target_lengths, blank
    )
    np_out = np.where(np_loss == float("inf"), 0, np_loss) if zero_infinity else np_loss
    if reduction == "mean":
        np_out = np.mean(
            np.divide(np_out, np.clip(target_lengths, 1, a_max=None).astype(np.float32))
        )
    elif reduction == "sum":
        np_out = np.sum(np_out)
    np_grad_out = np.ones_like(np_loss, dtype=np.float32)
    if reduction == "mean":
        np_grad_out = np.divide(
            np_grad_out, np.clip(target_lengths, 1, a_max=None).astype(np.float32)
        )
        np_grad_out /= target_lengths.size
    np_grad = ctc_loss_grad_np(
        np_grad_out,
        np_loss,
        np_alpha,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    )
    ctc_loss = flow.nn.CTCLoss(
        blank=blank, reduction=reduction, zero_infinity=zero_infinity
    )
    log_probs = flow.tensor(
        log_probs,
        dtype=flow.float32,
        requires_grad=True,
        device=flow.device(device_type),
    )
    targets = flow.tensor(
        targets, dtype=flow.int32, requires_grad=False, device=flow.device(device_type)
    )
    input_lengths = flow.tensor(
        input_lengths,
        dtype=flow.int32,
        requires_grad=False,
        device=flow.device(device_type),
    )
    target_lengths = flow.tensor(
        target_lengths,
        dtype=flow.int32,
        requires_grad=False,
        device=flow.device(device_type),
    )
    ctc_loss = ctc_loss.to(device_type)
    of_out = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    assert np.allclose(of_out.numpy(), np_out, atol=1e-05)
    of_out = of_out.sum()
    of_out.backward()
    assert np.allclose(log_probs.grad.numpy(), np_grad, atol=1e-05, equal_nan=True)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cuda", "cpu"]
    arg_dict["device_num"] = [1]
    arg_dict["data_type"] = ["float32"]
    arg_dict["max_input_length"] = [20]
    arg_dict["batch_size"] = [4]
    arg_dict["num_classes"] = [5]
    arg_dict["max_target_length"] = [10]
    arg_dict["blank"] = [0, 4]
    arg_dict["reduction"] = ["mean", "none"]
    arg_dict["zero_infinity"] = [False, True]
    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestCTCLoss1n1d(flow.unittest.TestCase):
    def test_ctc_loss(test_case):
        for arg in gen_arg_list():
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
