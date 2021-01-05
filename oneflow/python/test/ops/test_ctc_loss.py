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


def log_softmax(logits, axis=0):
    max_value = np.max(logits, axis, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis, keepdims=True)
    dist = exp / exp_sum
    return np.log(dist)


def ctc_loss_np(log_probs, targets, input_lengths, target_lengths, blank=0):
    ninf = -np.float("inf")

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

    def get_target_prime(targets, b, s, blank):
        if s % 2 == 0:
            return blank
        else:
            return targets[b, s // 2]

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
            target = get_target_prime(targets, b, 1, blank)
            alpha[b, 0, 1] = log_probs[0, b, target]

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


def compare_with_np(
    device_type, max_logit_length, batch_size, num_classes, max_label_length, data_type
):
    assert device_type in ["gpu", "cpu"]
    assert data_type in ["float32", "double", "int8", "int32", "int64"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def ctc_loss_job(
        log_probs: tp.Numpy.Placeholder(
            shape=(max_logit_length, batch_size, num_classes)
        ),
        targets: tp.Numpy.Placeholder(
            shape=(batch_size, max_label_length), dtype=flow.int32
        ),
        input_lengths: tp.Numpy.Placeholder(shape=(batch_size,), dtype=flow.int32),
        target_lengths: tp.Numpy.Placeholder(shape=(batch_size,), dtype=flow.int32),
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            return flow.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, reduction="none"
            )

    log_probs = np.random.random(
        size=(max_logit_length, batch_size, num_classes)
    ).astype(type_name_to_np_type[data_type])
    log_probs = log_softmax(log_probs, axis=2)

    targets = np.random.randint(
        1, high=num_classes, size=(batch_size, max_label_length)
    )
    input_lengths = np.random.randint(
        max_logit_length / 2, high=max_logit_length, size=(batch_size,)
    )
    target_lengths = np.random.randint(
        max_label_length / 2, high=max_label_length, size=(batch_size,)
    )

    # OneFlow
    of_out = ctc_loss_job(log_probs, targets, input_lengths, target_lengths)
    # Numpy
    np_out, _ = ctc_loss_np(log_probs, targets, input_lengths, target_lengths)

    tolerance = 1e-5
    assert np.allclose(of_out, np_out, rtol=tolerance, atol=tolerance)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["max_logit_length"] = [1000]
    arg_dict["batch_size"] = [10]
    arg_dict["num_classes"] = [10]
    arg_dict["max_label_length"] = [100]
    arg_dict["data_type"] = ["float32"]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestSort(flow.unittest.TestCase):
    def test_sort(test_case):
        for arg in gen_arg_list():
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
