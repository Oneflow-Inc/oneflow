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

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

import torch
import random


def _test_rnn(test_case, device):
    l = ["tanh", "relu"]
    input_size = random.randint(10, 1000)
    hidden_size = random.randint(10, 1000)
    num_layers = random.randint(1, 6)
    nonlinearity = l[0 if num_layers <= 3 else 1]
    bias = random.randint(-10, 10) <= 0
    batch_first = random.randint(-10, 10) <= 0
    dropout = 0
    bidirectional = random.randint(-10, 10) <= 0

    rnn_torch = torch.nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity=nonlinearity,
        bias=bias,
        batch_first=batch_first,
        dropout=0,
        bidirectional=bidirectional,
    ).to(device)

    weights_torch = []
    for w in rnn_torch.parameters():
        weights_torch.append(
            w.permute(1, 0).cpu().data.numpy()
            if len(w.size()) > 1
            else w.cpu().data.numpy()
        )

    rnn_flow = flow.nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity=nonlinearity,
        bias=bias,
        batch_first=batch_first,
        dropout=0,
        bidirectional=bidirectional,
    ).to(device)

    for i, w in enumerate(rnn_flow.parameters()):
        w_torch = weights_torch[i]
        w.copy_(flow.tensor(w_torch))

    x = np.random.rand(32, 10, input_size)
    x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    x_flow = flow.tensor(x, dtype=flow.float32, requires_grad=True).to(device)

    out_torch, hid_torch = rnn_torch(x_torch)
    out_flow, hid_flow = rnn_flow(x_flow)
    test_case.assertTrue(
        np.allclose(
            out_torch.cpu().data.numpy(),
            out_flow.cpu().data.numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
    )

    z_torch = out_torch.sum()
    z_torch.backward()
    z_flow = out_flow.sum()
    z_flow.backward()
    test_case.assertTrue(
        np.allclose(x_torch.cpu().data.numpy(), x_flow.cpu().data.numpy())
    )


def _test_lstm(test_case, device):
    input_size = random.randint(10, 1000)
    hidden_size = random.randint(12, 1000)
    num_layers = random.randint(1, 6)
    bias = random.randint(-10, 10) <= 0
    batch_first = random.randint(-10, 10) <= 0
    dropout = 0
    bidirectional = random.randint(-10, 10) <= 0
    proj_size = random.randint(10, hidden_size - 1)

    lstm_torch = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=0,
        bidirectional=bidirectional,
        proj_size=proj_size,
    ).to(device)

    weights_torch = []
    for w in lstm_torch.parameters():
        weights_torch.append(
            w.permute(1, 0).cpu().data.numpy()
            if len(w.size()) > 1
            else w.cpu().data.numpy()
        )

    lstm_flow = flow.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=0,
        bidirectional=bidirectional,
        proj_size=proj_size,
    ).to(device)

    for i, w in enumerate(lstm_flow.parameters()):
        w_torch = weights_torch[i]
        w.copy_(flow.tensor(w_torch))

    x = np.random.rand(32, 10, input_size)
    x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    x_flow = flow.tensor(x, dtype=flow.float32, requires_grad=True).to(device)

    out_torch, hid_torch = lstm_torch(x_torch)
    out_flow, hid_flow = lstm_flow(x_flow)
    test_case.assertTrue(
        np.allclose(
            out_torch.cpu().data.numpy(),
            out_flow.cpu().data.numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
    )

    z_torch = out_torch.sum()
    z_torch.backward()
    z_flow = out_flow.sum()
    z_flow.backward()
    test_case.assertTrue(
        np.allclose(x_torch.cpu().data.numpy(), x_flow.cpu().data.numpy())
    )


def _test_gru(test_case, device):
    input_size = random.randint(10, 1000)
    hidden_size = random.randint(10, 1000)
    num_layers = random.randint(1, 6)
    bias = bool(random.randint(-5, 5))
    batch_first = bool(random.randint(-5, 5))
    dropout = 0
    bidirectional = bool(random.randint(-5, 5))

    gru_torch = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=0,
        bidirectional=bidirectional,
    ).to(device)

    weights_torch = []
    for w in gru_torch.parameters():
        weights_torch.append(
            w.permute(1, 0).cpu().data.numpy()
            if len(w.size()) > 1
            else w.cpu().data.numpy()
        )

    gru_flow = flow.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=0,
        bidirectional=bidirectional,
    ).to(device)

    for i, w in enumerate(gru_flow.parameters()):
        w_torch = weights_torch[i]
        w.copy_(flow.tensor(w_torch))

    x = np.random.rand(32, 10, input_size)
    x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    x_flow = flow.tensor(x, dtype=flow.float32, requires_grad=True).to(device)

    out_torch, hid_torch = gru_torch(x_torch)
    out_flow, hid_flow = gru_flow(x_flow)
    test_case.assertTrue(
        np.allclose(
            out_torch.cpu().data.numpy(),
            out_flow.cpu().data.numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
    )

    z_torch = out_torch.sum()
    z_torch.backward()
    z_flow = out_flow.sum()
    z_flow.backward()
    test_case.assertTrue(
        np.allclose(x_torch.cpu().data.numpy(), x_flow.cpu().data.numpy())
    )


@flow.unittest.skip_unless_1n1d()
class TestRNNModule(flow.unittest.TestCase):
    def test_rnn(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_rnn, _test_lstm, _test_gru]
        arg_dict["device"] = ["cuda", "cpu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
