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


def _test_rnn_utils_pack_padded_sequence(test_case, device):
    input_size = random.randint(10, 150)
    max_seq_len = random.randint(10, 300)
    batch_size = random.randint(10, 300)
    requires_grad = np.random.rand() > 0.5
    padded_inputs = np.zeros((max_seq_len, batch_size, input_size))
    lengths = []
    lengths.append(max_seq_len)
    for i in range(batch_size - 1):
        lengths.append(random.randint(1, max_seq_len))
    lengths.sort(reverse=True)

    for i in range(batch_size):
        padded_inputs[0 : lengths[i], i : i + 1, :] = i + 1

    inputs = flow.from_numpy(padded_inputs).to(device)
    inputs.requires_grad = requires_grad
    flow_res = flow_rnn_utils.pack_padded_sequence(inputs, lengths)

    torch_inputs = torch.from_numpy(padded_inputs).to(device)
    torch_inputs.requires_grad = requires_grad
    torch_res = torch_rnn_utils.pack_padded_sequence(torch_inputs, lengths)

    test_case.assertTrue(
        np.allclose(
            torch_res.batch_sizes.cpu().detach().numpy(),
            flow_res.batch_sizes.cpu().detach().numpy(),
            atol=1e-8,
        )
    )

    test_case.assertTrue(
        np.allclose(
            torch_res.data.cpu().detach().numpy(),
            flow_res.data.cpu().detach().numpy(),
            atol=1e-8,
        )
    )

    torch_seq_unpacked, torch_lens_unpacked = torch_rnn_utils.pad_packed_sequence(
        torch_res, batch_first=False
    )
    flow_seq_unpacked, flow_lens_unpacked = flow_rnn_utils.pad_packed_sequence(
        flow_res, batch_first=False
    )

    if requires_grad:
        torch_seq_unpacked.sum().backward()
        flow_seq_unpacked.sum().backward()

    test_case.assertTrue(
        np.allclose(
            torch_seq_unpacked.cpu().detach().numpy(),
            flow_seq_unpacked.cpu().detach().numpy(),
            atol=1e-8,
        )
    )

    test_case.assertTrue(
        np.allclose(
            torch_lens_unpacked.cpu().detach().numpy(),
            flow_lens_unpacked.cpu().detach().numpy(),
            atol=1e-8,
        )
    )

    if requires_grad:
        test_case.assertTrue(
            np.allclose(inputs.grad.cpu().numpy(), torch_inputs.grad.cpu().numpy())
        )


def _test_rnn_utils_pad_sequence(test_case, device):
    input_size = random.randint(10, 150)
    max_seq_len = random.randint(20, 300)
    batch_size = random.randint(20, 300)
    lengths = []
    lengths.append(max_seq_len)
    for i in range(batch_size - 1):
        lengths.append(random.randint(1, max_seq_len))
    lengths.sort(reverse=True)

    sequences = []
    for i in range(batch_size):
        sequences.append(flow.rand(lengths[i], input_size).to(device))

    flow_res = flow_rnn_utils.pad_sequence(sequences)

    torch_inputs = [torch.tensor(ft.numpy(), device=device) for ft in sequences]
    torch_res = torch_rnn_utils.pad_sequence(torch_inputs)

    test_case.assertTrue(
        np.allclose(
            torch_res.cpu().detach().numpy(),
            flow_res.cpu().detach().numpy(),
            atol=1e-8,
        )
    )


def _test_rnn_utils_pack_sequence(test_case, device):
    input_size = random.randint(10, 150)
    max_seq_len = random.randint(20, 300)
    batch_size = random.randint(20, 300)
    lengths = []
    lengths.append(max_seq_len)
    for i in range(batch_size - 1):
        lengths.append(random.randint(1, max_seq_len))
    lengths.sort(reverse=True)

    sequences = []
    for i in range(batch_size):
        sequences.append(flow.rand(lengths[i], input_size).to(device))

    flow_res = flow_rnn_utils.pack_sequence(sequences)

    torch_inputs = [torch.tensor(ft.numpy(), device=device) for ft in sequences]
    torch_res = torch_rnn_utils.pack_sequence(torch_inputs)

    test_case.assertTrue(
        np.allclose(
            torch_res.batch_sizes.cpu().detach().numpy(),
            flow_res.batch_sizes.cpu().detach().numpy(),
            atol=1e-8,
        )
    )

    test_case.assertTrue(
        np.allclose(
            torch_res.data.cpu().detach().numpy(),
            flow_res.data.cpu().detach().numpy(),
            atol=1e-8,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestRNNUtils(flow.unittest.TestCase):
    def test_rnn_utils_pack_padded_sequence(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        for i in range(10):
            for arg in GenArgList(arg_dict):
                _test_rnn_utils_pack_padded_sequence(test_case, *arg[0:])

    def test_rnn_utils_pad_sequence(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        for i in range(10):
            for arg in GenArgList(arg_dict):
                _test_rnn_utils_pad_sequence(test_case, *arg[0:])

    def test_rnn_utils_pack_sequence(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        for i in range(10):
            for arg in GenArgList(arg_dict):
                _test_rnn_utils_pack_sequence(test_case, *arg[0:])


if __name__ == "__main__":
    unittest.main()
