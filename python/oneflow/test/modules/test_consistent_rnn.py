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

from collections import OrderedDict
import unittest

import torch
import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util.torch_flow_dual_object import global_view
from oneflow.test_utils.automated_test_util.generators import *
from test_util import GenArgDict


def test_rnn_impl(
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
    )
    for i, w in enumerate(rnn_flow.parameters()):
        w_torch = weights_torch[i]
        w.copy_(flow.tensor(w_torch))
    rnn_flow = rnn_flow.to_global(
        flow.env.all_device_placement("cpu"), flow.sbp.broadcast
    ).to_global(placement=placement, sbp=[module_sbp for _ in range(len(placement.hierarchy))])

    x = np.random.rand(32, 16, input_size).astype(np.float32)
    x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    x_flow = (
        flow.tensor(x, dtype=flow.float32, requires_grad=True)
        .to_global(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
        .to_global(placement=placement, sbp=in_sbp)
    )

    out_torch, hid_torch = rnn_torch(x_torch)
    out_flow, hid_flow = rnn_flow(x_flow)

    # check forward
    local_output = out_flow.to_global(placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(placement.hierarchy))]).to_local()
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
    local_x_grad = x_flow.to_global(placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(placement.hierarchy))]).to_local()
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.allclose(
                x_torch.cpu().detach().numpy(),
                local_x_grad.numpy(),
                rtol=1e-05,
                atol=1e-05,
            )
        )


class TestRNNConsistent(oneflow.unittest.TestCase):
    @global_view
    def test_rnn(test_case):
        arg_dict = OrderedDict()
        arg_dict["input_size"] = [
            80,
        ]
        arg_dict["hidden_size"] = [
            80,
        ]
        arg_dict["num_layers"] = [1, 3]
        arg_dict["nonlinearity"] = ["tanh", "relu"]
        arg_dict["bias"] = [True, False]
        arg_dict["batch_first"] = [True, False]
        arg_dict["dropout"] = [
            0,
        ]
        arg_dict["bidirectional"] = [True, False]

        # TODO: rnn module only support broadcast now
        module_sbp = flow.sbp.broadcast
        for args in GenArgDict(arg_dict):
            for placement in all_placement():
                for in_sbp in all_sbp(placement, max_dim=3):
                    # TODO: https://github.com/Oneflow-Inc/OneTeam/issues/1060
                    if flow.sbp.partial_sum in in_sbp:
                        continue
                    test_rnn_impl(test_case, placement, module_sbp, in_sbp, **args)


if __name__ == "__main__":
    unittest.main()
