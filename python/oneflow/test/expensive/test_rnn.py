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
class TestRNNModules(flow.unittest.TestCase):
    @autotest(n=5, check_graph=True, rtol=1e-2, atol=1e-3)
    def test_rnn(test_case):
        device = random_device()
        batch_size = random(1, 6)
        time_steps = random(1, 6)
        num_layers = random(1, 6).to(int)
        input_size = random(2, 6).to(int)
        hidden_size = random(2, 6).to(int)
        m = torch.nn.RNN(
            input_size,
            hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            bias=random().to(bool),
            batch_first=random().to(bool),
            dropout=0,
            bidirectional=random().to(bool),
        ).to(device)
        input = random_tensor(
            ndim=3, dim0=time_steps, dim1=batch_size, dim2=input_size
        ).to(device)
        out = m(input)
        return out[0]

    @autotest(n=5, check_graph=True, rtol=1e-2)
    def test_lstm(test_case):
        device = random_device()
        batch_size = random(1, 6)
        time_steps = random(1, 6)
        num_layers = random(1, 6).to(int)
        input_size = random(2, 6).to(int)
        hidden_size = random(2, 6).to(int)
        proj_size = random(2, 6).to(int)
        m = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=random().to(bool),
            batch_first=random().to(bool),
            dropout=0,
            bidirectional=random().to(bool),
            proj_size=proj_size,
        ).to(device)
        input = random_tensor(
            ndim=3, dim0=time_steps, dim1=batch_size, dim2=input_size
        ).to(device)
        out = m(input)
        return out[0]

    @autotest(n=5, check_graph=True, rtol=1e-2)
    def test_gru(test_case):
        device = random_device()
        batch_size = random(1, 6)
        time_steps = random(1, 6)
        num_layers = random(1, 6).to(int)
        input_size = random(2, 6).to(int)
        hidden_size = random(2, 6).to(int)
        m = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=random().to(bool),
            batch_first=random().to(bool),
            dropout=0,
            bidirectional=random().to(bool),
        ).to(device)
        input = random_tensor(
            ndim=3, dim0=time_steps, dim1=batch_size, dim2=input_size
        ).to(device)
        out = m(input)
        return out[0]


if __name__ == "__main__":
    unittest.main()
