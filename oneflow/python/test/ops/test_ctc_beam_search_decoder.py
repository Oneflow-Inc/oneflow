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
from collections import OrderedDict, defaultdict, Counter
from typing import Tuple


import numpy as np
import oneflow as flow
import oneflow.typing as tp
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import os
import tensorflow as tf

DEFAULT_BEAM_WIDTH = 10
DEFAULT_PRUNE_THRESHOLD = 0.001

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


def np_ctc_beam_search_decoder(log_probs, input_lengths, beam_width, top_paths):
    max_input_length, batch_size, num_classes = log_probs.shape
    # print(max_input_length, batch_size, num_classes)

    decoded = np.zeros(shape=(top_paths, batch_size, max_input_length), dtype=np.int32)
    log_probability = np.zeros(shape=(batch_size, top_paths), dtype=np.float32)

    for b in range(batch_size):
        Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
        Pb[0][()] = 1
        Pnb[0][()] = 0
        A_prev = [()]
        ctc = log_probs[:, b, :]
        ctc = np.vstack((np.zeros(num_classes), ctc))
        len_i = input_lengths[b] if input_lengths is not None else max_input_length

        for t in range(1, len_i + 1):
            pruned_alphabet = np.where(ctc[t] > DEFAULT_PRUNE_THRESHOLD)[0]
            # print(pruned_alphabet)
            for l in A_prev:
                for c in pruned_alphabet:
                    if c == 0:
                        Pb[t][l] += ctc[t][c] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        l_plus = l + (c,)
                        if len(l) > 0 and c == l[-1]:
                            Pnb[t][l_plus] += ctc[t][c] * Pb[t - 1][l]
                            Pnb[t][l] += ctc[t][c] * Pnb[t - 1][l]
                        else:
                            Pnb[t][l_plus] += ctc[t][c] * (Pb[t - 1][l] + Pnb[t - 1][l])

                        if l_plus not in A_prev:
                            Pb[t][l_plus] += ctc[t][0] * (
                                Pb[t - 1][l_plus] + Pnb[t - 1][l_plus]
                            )
                            Pnb[t][l_plus] += ctc[t][c] * Pnb[t - 1][l_plus]

            A_next = Pb[t] + Pnb[t]
            A_prev = sorted(A_next, key=A_next.get, reverse=True)
            A_prev = A_prev[:beam_width]

        candidates = A_prev[:top_paths]
        for p in range(len(candidates)):
            candidate = candidates[p]
            for c in range(len(candidate)):
                decoded[p, b, c] = candidate[c]
            log_probability[b, p] = Pb[t][candidate] + Pnb[t][candidate]
    return decoded, log_probability


def compare_with_np(
    device_type,
    device_num,
    data_type,
    max_input_length,
    batch_size,
    num_classes,
    beam_width=100,
    top_paths=3,
):
    assert data_type in ["float32", "double"]
    assert device_type in ["gpu", "cpu"]

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
    # log_probs = log_softmax(log_probs, axis=2)
    input_lengths = np.random.randint(
        max_input_length / 2, high=max_input_length, size=(batch_size,), dtype=np.int64
    )

    @flow.global_function(function_config=func_config)
    def ctc_beam_search_decoder_job(
        log_probs: tp.Numpy.Placeholder(
            shape=(max_input_length, batch_size, num_classes), dtype=flow_data_type
        ),
        input_lengths: tp.Numpy.Placeholder(shape=(batch_size,), dtype=flow.int64),
    ) -> Tuple[tp.Numpy, tp.Numpy]:
        with flow.scope.placement(device_type, "0:0"):
            decoded, neg_sum_logits = flow.nn.ctc_beam_search_decoder(
                log_probs, input_lengths, beam_width, top_paths
            )

        return decoded, neg_sum_logits

    of_decoded, of_neg_sum_logits = ctc_beam_search_decoder_job(
        log_probs, input_lengths
    )
    print(of_decoded)
    print(of_neg_sum_logits)

    np_decoded, np_neg_sum_logits = np_ctc_beam_search_decoder(
        log_probs, input_lengths, beam_width, top_paths
    )
    print(np_decoded)
    print(np_neg_sum_logits)

    tf_decoded, tf_neg_sum_logits = tf.nn.ctc_beam_search_decoder(
        log_probs, input_lengths
    )
    print(len(tf_decoded))
    for decoded in tf_decoded:
        print(tf.sparse.to_dense(decoded))
    print(tf_neg_sum_logits)

    # assert np.allclose(of_decoded, np_decoded, atol=1e-5)
    # assert np.allclose(of_neg_sum_logits, np_neg_sum_logits, atol=1e-5)


def gen_arg_list(type):
    arg_dict = OrderedDict()
    if type == "1n2d":
        arg_dict["device_type"] = ["gpu"]
        arg_dict["device_num"] = [2]
    else:
        arg_dict["device_type"] = ["cpu"]
        arg_dict["device_num"] = [1]
    arg_dict["data_type"] = ["float32"]
    arg_dict["max_input_length"] = [10]
    arg_dict["batch_size"] = [2]
    arg_dict["num_classes"] = [5]
    arg_dict["beam_width"] = [100]
    arg_dict["top_paths"] = [3]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestCTCBeamSearchDecoder1n1d(flow.unittest.TestCase):
    def test_ctc_beam_search_decoder(test_case):
        for arg in gen_arg_list("1n1d"):
            compare_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestCTCBeamSearchDecoder1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_ctc_beam_search_decoder(test_case):
        for arg in gen_arg_list("1n2d"):
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
