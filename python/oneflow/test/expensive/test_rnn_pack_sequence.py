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
import random
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.utils.rnn as torch_rnn_utils

import oneflow as flow
import oneflow.nn.utils.rnn as flow_rnn_utils
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList


def _test_rnn_pack_sequence(test_case, device):
    l = ["tanh", "relu"]
    input_size = random.randint(10, 1000)
    hidden_size = random.randint(10, 1000)
    num_layers = random.randint(1, 6)
    nonlinearity = l[0 if num_layers <= 3 else 1]
    grad_tol = 1e-4
    if nonlinearity == "relu":
        grad_tol = 100
    bias = random.randint(-10, 10) <= 0
    batch_first = False
    dropout = 0
    bidirectional = random.randint(-10, 10) <= 0

    rnn_torch = torch.nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity=nonlinearity,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    rnn_flow = flow.nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity=nonlinearity,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    torch_state_dict = rnn_torch.state_dict()
    new_dict = {}
    for k, v in torch_state_dict.items():
        new_dict[k] = v.detach().numpy()
    rnn_flow.load_state_dict(new_dict)

    rnn_flow = rnn_flow.to(device)
    rnn_torch = rnn_torch.to(device)

    max_seq_len = random.randint(10, 50)
    batch_size = random.randint(10, 50)
    lengths = []
    lengths.append(max_seq_len)
    for i in range(batch_size - 1):
        lengths.append(random.randint(1, max_seq_len))
    lengths.sort(reverse=True)

    sequences = []
    for i in range(batch_size):
        sequences.append(flow.rand(lengths[i], input_size).to(device))

    x_flow = flow_rnn_utils.pack_sequence(sequences)
    torch_inputs = [torch.tensor(ft.numpy(), device=device) for ft in sequences]
    x_torch = torch_rnn_utils.pack_sequence(torch_inputs)

    out_torch, hid_torch = rnn_torch(x_torch)
    out_flow, hid_flow = rnn_flow(x_flow)

    z_torch = out_torch.data.sum()
    z_torch.backward()
    z_flow = out_flow.data.sum()
    z_flow.backward()

    test_case.assertTrue(
        np.allclose(
            out_torch.data.cpu().detach().numpy(),
            out_flow.data.cpu().detach().numpy(),
            atol=1e-5,
        )
    )

    test_case.assertTrue(
        np.allclose(
            hid_torch.cpu().detach().numpy(),
            hid_flow.cpu().detach().numpy(),
            atol=1e-5,
        )
    )

    all_weights = rnn_torch.all_weights
    torch_params = []
    for ls in all_weights:
        for l in ls:
            torch_params.append(l)
    all_weights = rnn_flow.all_weights
    flow_params = []
    for ls in all_weights:
        for l in ls:
            flow_params.append(l)

    for i in range(len(flow_params)):
        torch_np = torch_params[i].grad.cpu().numpy()
        flow_np = flow_params[i].grad.cpu().numpy()
        test_case.assertTrue(np.allclose(torch_np, flow_np, atol=grad_tol))


def _test_lstm_pack_sequence(test_case, device):
    input_size = random.randint(10, 1000)
    hidden_size = random.randint(12, 1000)
    num_layers = random.randint(1, 6)
    bias = random.randint(-10, 10) <= 0
    batch_first = False
    dropout = 0
    bidirectional = random.randint(-10, 10) <= 0
    proj_size = random.randint(0, hidden_size - 1)

    lstm_torch = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
        proj_size=proj_size,
    )

    lstm_flow = flow.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
        proj_size=proj_size,
    )

    torch_state_dict = lstm_torch.state_dict()
    new_dict = {}
    for k, v in torch_state_dict.items():
        new_dict[k] = v.detach().numpy()
    lstm_flow.load_state_dict(new_dict)

    lstm_flow = lstm_flow.to(device)
    lstm_torch = lstm_torch.to(device)

    max_seq_len = random.randint(10, 50)
    batch_size = random.randint(10, 50)
    lengths = []
    lengths.append(max_seq_len)
    for i in range(batch_size - 1):
        lengths.append(random.randint(1, max_seq_len))
    lengths.sort(reverse=True)

    sequences = []
    for i in range(batch_size):
        sequences.append(flow.rand(lengths[i], input_size).to(device))

    x_flow = flow_rnn_utils.pack_sequence(sequences)
    torch_inputs = [torch.tensor(ft.numpy(), device=device) for ft in sequences]
    x_torch = torch_rnn_utils.pack_sequence(torch_inputs)

    out_torch, hid_torch = lstm_torch(x_torch)
    out_flow, hid_flow = lstm_flow(x_flow)

    z_torch = out_torch.data.sum()
    z_torch.backward()
    z_flow = out_flow.data.sum()
    z_flow.backward()

    test_case.assertTrue(
        np.allclose(
            out_torch.data.cpu().detach().numpy(),
            out_flow.data.cpu().detach().numpy(),
            atol=1e-5,
        )
    )

    test_case.assertTrue(
        np.allclose(
            hid_torch[0].cpu().detach().numpy(),
            hid_flow[0].cpu().detach().numpy(),
            atol=1e-5,
        )
    )

    test_case.assertTrue(
        np.allclose(
            hid_torch[1].cpu().detach().numpy(),
            hid_flow[1].cpu().detach().numpy(),
            atol=1e-5,
        )
    )

    all_weights = lstm_torch.all_weights
    torch_params = []
    for ls in all_weights:
        for l in ls:
            torch_params.append(l)
    all_weights = lstm_flow.all_weights
    flow_params = []
    for ls in all_weights:
        for l in ls:
            flow_params.append(l)

    for i in range(len(flow_params)):
        torch_np = torch_params[i].grad.cpu().numpy()
        flow_np = flow_params[i].grad.cpu().numpy()
        test_case.assertTrue(np.allclose(torch_np, flow_np, atol=1e-4))


def _test_gru_pack_sequence(test_case, device):
    input_size = random.randint(10, 1000)
    hidden_size = random.randint(10, 1000)
    num_layers = random.randint(1, 6)
    grad_tol = 1e-4
    bias = random.randint(-10, 10) <= 0
    batch_first = False
    dropout = 0
    bidirectional = random.randint(-10, 10) <= 0

    gru_torch = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    gru_flow = flow.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    torch_state_dict = gru_torch.state_dict()
    new_dict = {}
    for k, v in torch_state_dict.items():
        new_dict[k] = v.detach().numpy()
    gru_flow.load_state_dict(new_dict)

    gru_flow = gru_flow.to(device)
    gru_torch = gru_torch.to(device)

    max_seq_len = random.randint(10, 50)
    batch_size = random.randint(10, 50)
    lengths = []
    lengths.append(max_seq_len)
    for i in range(batch_size - 1):
        lengths.append(random.randint(1, max_seq_len))
    lengths.sort(reverse=True)

    sequences = []
    for i in range(batch_size):
        sequences.append(flow.rand(lengths[i], input_size).to(device))

    x_flow = flow_rnn_utils.pack_sequence(sequences)
    torch_inputs = [torch.tensor(ft.numpy(), device=device) for ft in sequences]
    x_torch = torch_rnn_utils.pack_sequence(torch_inputs)

    out_torch, hid_torch = gru_torch(x_torch)
    out_flow, hid_flow = gru_flow(x_flow)

    z_torch = out_torch.data.sum()
    z_torch.backward()
    z_flow = out_flow.data.sum()
    z_flow.backward()

    test_case.assertTrue(
        np.allclose(
            out_torch.data.cpu().detach().numpy(),
            out_flow.data.cpu().detach().numpy(),
            atol=1e-5,
        )
    )

    test_case.assertTrue(
        np.allclose(
            hid_torch.cpu().detach().numpy(),
            hid_flow.cpu().detach().numpy(),
            atol=1e-5,
        )
    )

    all_weights = gru_torch.all_weights
    torch_params = []
    for ls in all_weights:
        for l in ls:
            torch_params.append(l)
    all_weights = gru_flow.all_weights
    flow_params = []
    for ls in all_weights:
        for l in ls:
            flow_params.append(l)

    for i in range(len(flow_params)):
        torch_np = torch_params[i].grad.cpu().numpy()
        flow_np = flow_params[i].grad.cpu().numpy()
        test_case.assertTrue(np.allclose(torch_np, flow_np, atol=grad_tol))


@flow.unittest.skip_unless_1n1d()
class TestRNNModules(flow.unittest.TestCase):
    def test_rnn(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_rnn_pack_sequence,
            _test_lstm_pack_sequence,
            _test_gru_pack_sequence,
        ]
        arg_dict["device"] = ["cuda", "cpu"]
        for i in range(5):
            for arg in GenArgList(arg_dict):
                arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
