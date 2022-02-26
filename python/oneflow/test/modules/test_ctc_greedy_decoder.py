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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

ninf = -float("inf")


def log_softmax(logits, axis=0):
    max_value = np.max(logits, axis, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis, keepdims=True)
    dist = exp / exp_sum
    return np.log(dist)


def np_ctc_greedy_decoder(log_probs, input_lengths, merge_repeated=True):
    blank_label = log_probs.shape[2] - 1
    decodes = np.zeros(
        (log_probs.shape[1], log_probs.shape[0]), dtype=input_lengths.dtype
    )
    neg_sum_logits = np.zeros((input_lengths.size, 1), dtype=log_probs.dtype)
    for b in range(input_lengths.size):
        input_length = input_lengths[b]
        prev_indices = -1
        t_dec = 0
        for t in range(input_length):
            max_indice = np.argmax(log_probs[t, b, :])
            neg_sum_logits[b, 0] -= log_probs[t, b, max_indice]
            if max_indice != blank_label and (
                not (merge_repeated and max_indice == prev_indices)
            ):
                decodes[b, t_dec] = max_indice
                t_dec += 1
            prev_indices = max_indice
    return (decodes, neg_sum_logits)


def compare_with_np(
    device_type, data_type, max_input_length, batch_size, num_classes, merge_repeated,
):
    assert data_type in ["float32", "double"]
    assert device_type in ["cpu", "cuda"]
    assert merge_repeated in [False, True]

    log_probs = np.random.random(
        size=(max_input_length, batch_size, num_classes)
    ).astype(np.float32)
    log_probs = log_softmax(log_probs, axis=2)
    input_lengths = np.random.randint(
        max_input_length / 2, high=max_input_length, size=(batch_size,), dtype=np.int64
    )
    (np_decoded, np_neg_sum_logits) = np_ctc_greedy_decoder(
        log_probs, input_lengths, merge_repeated
    )

    log_probs = flow.tensor(
        log_probs,
        dtype=flow.float32,
        requires_grad=False,
        device=flow.device(device_type),
    )

    input_lengths = flow.tensor(
        input_lengths,
        dtype=flow.int64,
        requires_grad=False,
        device=flow.device(device_type),
    )

    (of_decoded, of_neg_sum_logits) = flow.nn.functional.ctc_greedy_decoder(
        log_probs, input_lengths, merge_repeated
    )
    np.allclose(of_decoded.numpy(), np_decoded, atol=1e-05)
    np.allclose(of_neg_sum_logits.numpy(), np_neg_sum_logits, atol=1e-05)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "cuda"]
    arg_dict["data_type"] = ["float32"]
    arg_dict["max_input_length"] = [20]
    arg_dict["batch_size"] = [4]
    arg_dict["num_classes"] = [5]
    arg_dict["merge_repeated"] = [False, True]
    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestCTCGreedyDecoder1n1d(flow.unittest.TestCase):
    def test_ctc_greedy_decoder(test_case):
        for arg in gen_arg_list():
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
