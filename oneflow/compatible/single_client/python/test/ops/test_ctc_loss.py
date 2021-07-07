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
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as tp
import os

ninf = -float("inf")


def _logsumexp(a, b):
    if a < b:
        a, b = b, a
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

    max_input_length, batch_size, _ = log_probs.shape
    _, max_target_length = targets.shape
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
    return loss, alpha


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
    max_input_length, batch_size, num_labels = log_probs.shape
    _, max_target_length = targets.shape

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
    assert device_type in ["gpu", "cpu"]
    assert reduction in ["none", "mean", "sum"]
    assert zero_infinity in [False, True]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_num)
    else:
        flow.config.gpu_device_num(device_num)
    flow_data_type = type_name_to_flow_type[data_type]
    np_data_type = type_name_to_np_type[data_type]
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow_data_type)
    func_config.default_placement_scope(
        flow.scope.placement(device_type, "0:0-{}".format(device_num - 1))
    )

    log_probs = np.random.random(
        size=(max_input_length, batch_size, num_classes)
    ).astype(np_data_type)
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

    np_loss, np_alpha = ctc_loss_np(
        log_probs, targets, input_lengths, target_lengths, blank
    )

    np_out = np.where(np_loss == float("inf"), 0, np_loss) if zero_infinity else np_loss
    if reduction == "mean":
        np_out = np.mean(
            np.divide(
                np_out, np.clip(target_lengths, 1, a_max=None).astype(np_data_type)
            )
        )
    elif reduction == "sum":
        np_out = np.sum(np_out)

    np_grad_out = np.ones_like(np_loss, dtype=np_data_type)
    if reduction == "mean":
        np_grad_out = np.divide(
            np_grad_out, np.clip(target_lengths, 1, a_max=None).astype(np_data_type)
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

    def assert_loss_grad(blob: tp.Numpy):
        assert np.allclose(blob, np_grad, atol=1e-5, equal_nan=True)

    @flow.global_function(type="train", function_config=func_config)
    def ctc_loss_job(
        log_probs: tp.Numpy.Placeholder(
            shape=(max_input_length, batch_size, num_classes), dtype=flow_data_type
        ),
        targets: tp.Numpy.Placeholder(
            shape=(batch_size, max_target_length), dtype=flow.int32
        ),
        input_lengths: tp.Numpy.Placeholder(shape=(batch_size,), dtype=flow.int32),
        target_lengths: tp.Numpy.Placeholder(shape=(batch_size,), dtype=flow.int32),
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=log_probs.shape,
                dtype=flow_data_type,
                initializer=flow.zeros_initializer(),
                name="x_var",
            )
            x_var = log_probs + v

        flow.watch_diff(x_var, assert_loss_grad)
        loss = flow.ctc_loss(
            x_var,
            targets,
            input_lengths,
            target_lengths,
            blank,
            reduction,
            zero_infinity,
        )

        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(loss)

        return loss

    of_out = ctc_loss_job(log_probs, targets, input_lengths, target_lengths)
    assert np.allclose(of_out, np_out, atol=1e-5)


def gen_arg_list(type):
    arg_dict = OrderedDict()
    if type == "1n2d":
        arg_dict["device_type"] = ["gpu"]
        arg_dict["device_num"] = [2]
    else:
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["device_num"] = [1]
    arg_dict["data_type"] = ["float32"]
    arg_dict["max_input_length"] = [20]
    arg_dict["batch_size"] = [4]
    arg_dict["num_classes"] = [5]
    arg_dict["max_target_length"] = [10]
    arg_dict["blank"] = [0, 4]  # 0 <= blank < num_classes
    arg_dict["reduction"] = ["mean", "none"]
    arg_dict["zero_infinity"] = [False, True]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestCTCLoss1n1d(flow.unittest.TestCase):
    def test_ctc_loss(test_case):
        for arg in gen_arg_list("1n1d"):
            compare_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestCTCLoss1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_ctc_loss(test_case):
        for arg in gen_arg_list("1n2d"):
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
