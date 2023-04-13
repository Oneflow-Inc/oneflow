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
import oneflow.unittest
import torch
from oneflow.test_utils.automated_test_util.generators import *
from oneflow.test_utils.automated_test_util.torch_flow_dual_object import globaltest
from oneflow.test_utils.test_util import GenArgDict


def _compare_torch_and_oneflow(
    test_case, m_torch, m_flow, placement, module_sbp, in_sbp, input_size
):
    torch_state_dict = m_torch.state_dict()
    new_dict = {}
    for k, v in torch_state_dict.items():
        new_dict[k] = v.detach().numpy()
    m_flow.load_state_dict(new_dict)

    m_flow = m_flow.to_global(flow.placement.all("cpu"), flow.sbp.broadcast).to_global(
        placement=placement, sbp=[module_sbp for _ in range(len(placement.ranks.shape))]
    )

    x = np.random.rand(32, 16, input_size).astype(np.float32)
    x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    x_flow = (
        flow.tensor(x, dtype=flow.float32, requires_grad=True)
        .to_global(flow.placement.all("cpu"), flow.sbp.broadcast)
        .to_global(placement=placement, sbp=in_sbp)
    )

    out_torch, hid_torch = m_torch(x_torch)
    out_flow, hid_flow = m_flow(x_flow)

    # check forward
    local_output = out_flow.to_global(
        placement=placement,
        sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))],
    ).to_local()
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.allclose(
                out_torch.cpu().detach().numpy(),
                local_output.numpy(),
                rtol=1e-05,
                atol=1e-05,
            )
        )

    # check backward
    out_torch.sum().backward()
    out_flow.sum().backward()
    local_x_grad = x_flow.to_global(
        placement=placement,
        sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))],
    ).to_local()
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.allclose(
                x_torch.cpu().detach().numpy(),
                local_x_grad.numpy(),
                rtol=1e-05,
                atol=1e-05,
            )
        )


def _test_rnn_impl(
    test_case,
    placement,
    module_sbp,
    in_sbp,
    input_size,
    hidden_size,
    num_layers,
    nonlinearity,
    bias,
    batch_first,
    dropout,
    bidirectional,
):
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
    _compare_torch_and_oneflow(
        test_case, rnn_torch, rnn_flow, placement, module_sbp, in_sbp, input_size
    )


def _test_lstm_impl(
    test_case,
    placement,
    module_sbp,
    in_sbp,
    input_size,
    hidden_size,
    num_layers,
    bias,
    batch_first,
    dropout,
    bidirectional,
    proj_size,
):
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
    _compare_torch_and_oneflow(
        test_case, lstm_torch, lstm_flow, placement, module_sbp, in_sbp, input_size
    )


def _test_gru_impl(
    test_case,
    placement,
    module_sbp,
    in_sbp,
    input_size,
    hidden_size,
    num_layers,
    bias,
    batch_first,
    dropout,
    bidirectional,
):
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
    _compare_torch_and_oneflow(
        test_case, gru_torch, gru_flow, placement, module_sbp, in_sbp, input_size
    )


class TestRNNGlobal(oneflow.unittest.TestCase):
    @globaltest
    def test_rnn(test_case):
        arg_dict = OrderedDict()
        arg_dict["input_size"] = [
            1,
        ]
        arg_dict["hidden_size"] = [
            1,
        ]
        arg_dict["num_layers"] = [
            1,
        ]
        arg_dict["nonlinearity"] = ["tanh", "relu"]
        arg_dict["bias"] = [True, False]
        arg_dict["batch_first"] = [True, False]
        arg_dict["dropout"] = [
            0,
        ]
        arg_dict["bidirectional"] = [True, False]

        module_sbp = flow.sbp.broadcast
        for args in GenArgDict(arg_dict):
            for placement in all_placement():
                for in_sbp in all_sbp(placement, max_dim=3, valid_split_axis=1):
                    _test_rnn_impl(test_case, placement, module_sbp, in_sbp, **args)

    @globaltest
    def test_lstm(test_case):
        arg_dict = OrderedDict()
        arg_dict["input_size"] = [
            1,
        ]
        arg_dict["hidden_size"] = [
            2,
        ]
        arg_dict["num_layers"] = [
            1,
        ]
        arg_dict["bias"] = [True, False]
        arg_dict["batch_first"] = [True, False]
        arg_dict["dropout"] = [
            0,
        ]
        arg_dict["bidirectional"] = [True, False]
        arg_dict["proj_size"] = [0, 1]

        module_sbp = flow.sbp.broadcast
        for args in GenArgDict(arg_dict):
            for placement in all_placement():
                for in_sbp in all_sbp(placement, max_dim=3, valid_split_axis=1):
                    _test_lstm_impl(test_case, placement, module_sbp, in_sbp, **args)

    @globaltest
    def test_gru(test_case):
        arg_dict = OrderedDict()
        arg_dict["input_size"] = [
            1,
        ]
        arg_dict["hidden_size"] = [
            1,
        ]
        arg_dict["num_layers"] = [
            1,
        ]
        arg_dict["bias"] = [True, False]
        arg_dict["batch_first"] = [True, False]
        arg_dict["dropout"] = [
            0,
        ]
        arg_dict["bidirectional"] = [True, False]

        module_sbp = flow.sbp.broadcast
        for args in GenArgDict(arg_dict):
            for placement in all_placement():
                for in_sbp in all_sbp(placement, max_dim=3, valid_split_axis=1):
                    _test_gru_impl(test_case, placement, module_sbp, in_sbp, **args)


if __name__ == "__main__":
    unittest.main()
