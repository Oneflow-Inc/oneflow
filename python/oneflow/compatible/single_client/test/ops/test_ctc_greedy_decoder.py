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

import os
import unittest
from collections import OrderedDict
from typing import Tuple

import numpy as np
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp

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
    device_type,
    device_num,
    data_type,
    max_input_length,
    batch_size,
    num_classes,
    merge_repeated,
):
    assert data_type in ["float32", "double"]
    assert device_type in ["gpu", "cpu"]
    assert merge_repeated in [False, True]
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
    input_lengths = np.random.randint(
        max_input_length / 2, high=max_input_length, size=(batch_size,), dtype=np.int64
    )

    @flow.global_function(function_config=func_config)
    def ctc_greedy_decoder_job(
        log_probs: tp.Numpy.Placeholder(
            shape=(max_input_length, batch_size, num_classes), dtype=flow_data_type
        ),
        input_lengths: tp.Numpy.Placeholder(shape=(batch_size,), dtype=flow.int64),
    ) -> Tuple[tp.Numpy, tp.Numpy]:
        with flow.scope.placement(device_type, "0:0"):
            (decoded, neg_sum_logits) = flow.nn.ctc_greedy_decoder(
                log_probs, input_lengths, merge_repeated
            )
        return (decoded, neg_sum_logits)

    (of_decoded, of_neg_sum_logits) = ctc_greedy_decoder_job(log_probs, input_lengths)
    (np_decoded, np_neg_sum_logits) = np_ctc_greedy_decoder(
        log_probs, input_lengths, merge_repeated
    )
    np.allclose(of_decoded, np_decoded, atol=1e-05)
    np.allclose(of_neg_sum_logits, np_neg_sum_logits, atol=1e-05)


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
    arg_dict["merge_repeated"] = [False, True]
    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestCTCGreedyDecoder1n1d(flow.unittest.TestCase):
    def test_ctc_greedy_decoder(test_case):
        for arg in gen_arg_list("1n1d"):
            compare_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestCTCGreedyDecoder1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_ctc_greedy_decoder(test_case):
        for arg in gen_arg_list("1n2d"):
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
