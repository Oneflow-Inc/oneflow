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
import oneflow as flow

import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestRNN(flow.unittest.TestCase):
    @autotest(n=5, check_graph=True, rtol=1e-2, atol=1e-3)
    def test_rnn_tanh_cell(test_case):
        device = random_device()
        batch_size = random(1, 6)
        time_steps = random(1, 6)
        input_size = random(1, 6) * 2
        hidden_size = random(1, 6) * 2
        m = torch.nn.RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=random().to(bool),
            nonlinearity="tanh",
        ).to(device)
        input = random_tensor(
            ndim=3, dim0=time_steps, dim1=batch_size, dim2=input_size
        ).to(device)
        hx = random_tensor(ndim=2, dim0=batch_size, dim1=hidden_size).to(device)
        for i in range(time_steps.to(int).value()):
            hx = m(input[i], hx)
        return hx

    @autotest(n=5, check_graph=True)
    def test_rnn_relu_cell(test_case):
        device = random_device()
        batch_size = random(1, 6)
        time_steps = random(1, 6)
        input_size = random(1, 6) * 2
        hidden_size = random(1, 6) * 2
        m = torch.nn.RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=random().to(bool),
            nonlinearity="relu",
        ).to(device)
        input = random_tensor(
            ndim=3, dim0=time_steps, dim1=batch_size, dim2=input_size
        ).to(device)
        hx = random_tensor(ndim=2, dim0=batch_size, dim1=hidden_size).to(device)
        for i in range(time_steps.to(int).value()):
            hx = m(input[i], hx)
        return hx

    @autotest(n=5, check_graph=True, rtol=1e-2)
    def test_lstm_cell(test_case):
        device = random_device()
        batch_size = random(1, 6)
        time_steps = random(1, 6)
        input_size = random(1, 6) * 2
        hidden_size = random(1, 6) * 2
        has_bias = random().to(bool)
        cx_requires_grad = random().to(bool)
        m = torch.nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, bias=has_bias,
        ).to(device)
        input = random_tensor(
            ndim=3, dim0=time_steps, dim1=batch_size, dim2=input_size
        ).to(device)
        hx = random_tensor(
            ndim=2, dim0=batch_size, dim1=hidden_size, requires_grad=False
        ).to(device)
        cx = random_tensor(
            ndim=2, dim0=batch_size, dim1=hidden_size, requires_grad=cx_requires_grad
        ).to(device)

        for i in range(time_steps.to(int).value()):
            res = m(input[i], (hx, cx))
            hx = res[0]
            cx = res[1]
        return res[0]

    @autotest(n=5, check_graph=True, rtol=1e-2)
    def test_gru_cell(test_case):
        device = random_device()
        batch_size = random(1, 6)
        time_steps = random(1, 6)
        input_size = random(1, 6) * 2
        hidden_size = random(1, 6) * 2
        has_bias = random().to(bool)
        m = torch.nn.GRUCell(
            input_size=input_size, hidden_size=hidden_size, bias=has_bias
        ).to(device)
        input = random_tensor(
            ndim=3, dim0=time_steps, dim1=batch_size, dim2=input_size
        ).to(device)
        hx = random_tensor(ndim=2, dim0=batch_size, dim1=hidden_size).to(device)
        for i in range(time_steps.to(int).value()):
            hx = m(input[i], hx)
        return hx


if __name__ == "__main__":
    unittest.main()
