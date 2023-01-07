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


@autotest(n=1, check_graph=False)
def _test_rnn_relu_cell(test_case, placement, sbp):
    batch_size = random(2, 3) * 8
    time_steps = random(2, 3) * 8
    input_size = random(2, 3) * 8
    hidden_size = random(2, 3) * 8
    has_bias = random().to(bool)
    m = torch.nn.RNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=has_bias,
        nonlinearity="relu",
    )

    weight_sbp = random_sbp(placement, max_dim=2, except_partial_sum=True)
    m.weight_ih = torch.nn.Parameter(
        m.weight_ih.to_global(placement=placement, sbp=weight_sbp)
    )
    m.weight_hh = torch.nn.Parameter(
        m.weight_hh.to_global(placement=placement, sbp=weight_sbp)
    )
    if m.bias_ih is not None:
        # bias is 1-d tensor
        bias_sbp = random_sbp(placement, max_dim=1, except_partial_sum=True)
        m.bias_ih = torch.nn.Parameter(
            m.bias_ih.to_global(placement=placement, sbp=bias_sbp)
        )
        m.bias_hh = torch.nn.Parameter(
            m.bias_hh.to_global(placement=placement, sbp=bias_sbp)
        )

    input_sbp = random_sbp(placement, max_dim=3, valid_split_axis=1)
    input = random_tensor(
        ndim=3, dim0=time_steps, dim1=batch_size, dim2=input_size
    ).to_global(placement=placement, sbp=input_sbp)
    hx = random_tensor(
        ndim=2, dim0=batch_size, dim1=hidden_size, requires_grad=False
    ).to_global(placement=placement, sbp=sbp)

    for i in range(time_steps.to(int).value()):
        hx = m(input[i], hx)

    return hx


@autotest(n=1, check_graph=False)
def _test_rnn_tanh_cell(test_case, placement, sbp):
    batch_size = random(2, 3) * 8
    time_steps = random(2, 3) * 8
    input_size = random(2, 3) * 8
    hidden_size = random(2, 3) * 8
    has_bias = random().to(bool)
    m = torch.nn.RNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=has_bias,
        nonlinearity="tanh",
    )

    weight_sbp = random_sbp(placement, max_dim=2, except_partial_sum=True)
    m.weight_ih = torch.nn.Parameter(
        m.weight_ih.to_global(placement=placement, sbp=weight_sbp)
    )
    m.weight_hh = torch.nn.Parameter(
        m.weight_hh.to_global(placement=placement, sbp=weight_sbp)
    )
    if m.bias_ih is not None:
        # bias is 1-d tensor
        bias_sbp = random_sbp(placement, max_dim=1, except_partial_sum=True)
        m.bias_ih = torch.nn.Parameter(
            m.bias_ih.to_global(placement=placement, sbp=bias_sbp)
        )
        m.bias_hh = torch.nn.Parameter(
            m.bias_hh.to_global(placement=placement, sbp=bias_sbp)
        )

    input_sbp = random_sbp(placement, max_dim=3, valid_split_axis=1)
    input = random_tensor(
        ndim=3, dim0=time_steps, dim1=batch_size, dim2=input_size
    ).to_global(placement=placement, sbp=input_sbp)
    hx = random_tensor(
        ndim=2, dim0=batch_size, dim1=hidden_size, requires_grad=False
    ).to_global(placement=placement, sbp=sbp)

    for i in range(time_steps.to(int).value()):
        hx = m(input[i], hx)

    return hx


class TestRNNCellGlobal(flow.unittest.TestCase):
    @globaltest
    def test_rnn_relu_cell(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_rnn_relu_cell(test_case, placement, sbp)

    @globaltest
    def test_rnn_tanh_cell(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_rnn_tanh_cell(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
