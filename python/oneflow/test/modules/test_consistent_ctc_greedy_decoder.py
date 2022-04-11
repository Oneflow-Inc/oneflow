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
import numpy as np

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def log_softmax(logits, axis=0):
    max_value = flow.max(logits, axis, keepdim=True)[0]
    exp = flow.exp(logits - max_value)
    exp_sum = flow.sum(exp, axis, keepdim=True)
    dist = exp / exp_sum
    return flow.log(dist)


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


def compare_with_np(test_case, placement, sbp):
    max_input_length = random(2, 4).to(int).value() * 8
    batch_size = random(2, 4).to(int).value() * 8
    num_classes = random(2, 10).to(int).value()
    merge_repeated = random_bool().value()

    log_probs = random_tensor(
        ndim=3,
        dim0=max_input_length,
        dim1=batch_size,
        dim2=num_classes,
        requires_grad=False,
    ).oneflow
    log_probs = log_probs.to_global(placement=placement, sbp=sbp)
    log_probs = log_softmax(log_probs, axis=2)

    input_lengths = random_tensor(
        ndim=1,
        dim0=batch_size,
        low=max_input_length / 2,
        high=max_input_length,
        dtype=int,
        requires_grad=False,
    ).oneflow
    input_lengths = input_lengths.to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=1).value()
    )

    (np_decoded, np_neg_sum_logits) = np_ctc_greedy_decoder(
        log_probs.numpy(), input_lengths.numpy(), merge_repeated
    )

    (of_decoded, of_neg_sum_logits) = flow.nn.functional.ctc_greedy_decoder(
        log_probs, input_lengths, merge_repeated
    )
    test_case.assertTrue(np.allclose(of_decoded.numpy(), np_decoded, atol=1e-05))
    # test_case.assertTrue(np.allclose(of_neg_sum_logits.numpy(), np_neg_sum_logits, atol=1e-05))


class TestConsistentCTCGreedyDecoder1n1d(flow.unittest.TestCase):
    @globaltest
    def test_ctc_greedy_decoder(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                compare_with_np(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
