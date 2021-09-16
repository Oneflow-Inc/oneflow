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
from collections import OrderedDict

import torch
import numpy as np
from test_util import GenArgList


def _test_convtranspose1d_bias_false(test_case, device):
    np_arr = np.random.randn(1, 1, 3)
    weight = np.ones((1, 2, 3))

    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = flow.nn.ConvTranspose1d(1, 2, 3, stride=1, bias=False)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)

    input_torch = torch.tensor(np_arr, dtype=torch.float32, requires_grad=True)
    m_t = torch.nn.ConvTranspose1d(1, 2, 3, stride=1, bias=False)
    m_t.weight.data = torch.tensor(weight, dtype=torch.float32)
    out_torch = m_t(input_torch)

    test_case.assertTrue(
        np.allclose(out_flow.numpy(), out_torch.detach().numpy(), 1e-06, 1e-06)
    )
    out_flow = out_flow.sum()
    out_flow.backward()
    out_torch = out_torch.sum()
    out_torch.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), input_torch.grad.numpy(), 1e-06, 1e-06)
    )


def _test_convtranspose1d_bias_true(test_case, device):
    np_arr = np.random.randn(1, 1, 3)
    weight = np.ones((1, 2, 3))
    bias = np.random.rand(2)

    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = flow.nn.ConvTranspose1d(1, 2, 3, stride=1, bias=True)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f.bias = flow.nn.Parameter(flow.Tensor(bias))
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)

    input_torch = torch.tensor(np_arr, dtype=torch.float32, requires_grad=True)
    m_t = torch.nn.ConvTranspose1d(1, 2, 3, stride=1, bias=True)
    m_t.weight.data = torch.tensor(weight, dtype=torch.float32)
    m_t.bias = torch.nn.Parameter(torch.Tensor(bias))
    out_torch = m_t(input_torch)

    test_case.assertTrue(
        np.allclose(out_flow.numpy(), out_torch.detach().numpy(), 1e-06, 1e-06)
    )
    out_flow = out_flow.sum()
    out_flow.backward()
    out_torch = out_torch.sum()
    out_torch.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), input_torch.grad.numpy(), 1e-06, 1e-06)
    )


def _test_convtranspose1d_group_bias_false(test_case, device):
    np_arr = np.random.randn(2, 2, 3)
    weight = np.ones((2, 1, 3))

    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = flow.nn.ConvTranspose1d(2, 2, 3, stride=1, groups=2, bias=False)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)

    input_torch = torch.tensor(np_arr, dtype=torch.float32, requires_grad=True)
    m_t = torch.nn.ConvTranspose1d(2, 2, 3, stride=1, groups=2, bias=False)
    m_t.weight.data = torch.tensor(weight, dtype=torch.float32)
    out_torch = m_t(input_torch)

    test_case.assertTrue(
        np.allclose(out_flow.numpy(), out_torch.detach().numpy(), 1e-06, 1e-06)
    )
    out_flow = out_flow.sum()
    out_flow.backward()
    out_torch = out_torch.sum()
    out_torch.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), input_torch.grad.numpy(), 1e-06, 1e-06)
    )


def _test_convtranspose1d_group_bias_true(test_case, device):
    np_arr = np.random.randn(2, 2, 3)
    weight = np.ones((2, 1, 3))
    bias = np.random.rand(2)

    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = flow.nn.ConvTranspose1d(2, 2, 3, stride=1, groups=2, bias=True)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f.bias = flow.nn.Parameter(flow.Tensor(bias))
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)

    input_torch = torch.tensor(np_arr, dtype=torch.float32, requires_grad=True)
    m_t = torch.nn.ConvTranspose1d(2, 2, 3, stride=1, groups=2, bias=True)
    m_t.weight.data = torch.tensor(weight, dtype=torch.float32)
    m_t.bias = torch.nn.Parameter(torch.Tensor(bias))
    out_torch = m_t(input_torch)

    test_case.assertTrue(
        np.allclose(out_flow.numpy(), out_torch.detach().numpy(), 1e-06, 1e-06)
    )
    out_flow = out_flow.sum()
    out_flow.backward()
    out_torch = out_torch.sum()
    out_torch.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), input_torch.grad.numpy(), 1e-06, 1e-06)
    )


def _test_convtranspose1d_group_large_out_channel(test_case, device):
    np_arr = np.random.randn(2, 2, 3)
    weight = np.ones((2, 3, 3))

    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = flow.nn.ConvTranspose1d(2, 6, 3, stride=1, groups=2, bias=False)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)

    input_torch = torch.tensor(np_arr, dtype=torch.float32, requires_grad=True)
    m_t = torch.nn.ConvTranspose1d(2, 6, 3, stride=1, groups=2, bias=False)
    m_t.weight.data = torch.tensor(weight, dtype=torch.float32)
    out_torch = m_t(input_torch)

    test_case.assertTrue(
        np.allclose(out_flow.numpy(), out_torch.detach().numpy(), 1e-06, 1e-06)
    )
    out_flow = out_flow.sum()
    out_flow.backward()
    out_torch = out_torch.sum()
    out_torch.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), input_torch.grad.numpy(), 1e-06, 1e-06)
    )


def _test_convtranspose1d_group_large_in_channel(test_case, device):
    np_arr = np.random.randn(2, 4, 3)
    weight = np.ones((4, 1, 3))

    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = flow.nn.ConvTranspose1d(4, 2, 3, stride=1, groups=2, bias=False)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)

    input_torch = torch.tensor(np_arr, dtype=torch.float32, requires_grad=True)
    m_t = torch.nn.ConvTranspose1d(4, 2, 3, stride=1, groups=2, bias=False)
    m_t.weight.data = torch.tensor(weight, dtype=torch.float32)
    out_torch = m_t(input_torch)

    test_case.assertTrue(
        np.allclose(out_flow.numpy(), out_torch.detach().numpy(), 1e-06, 1e-06)
    )
    out_flow = out_flow.sum()
    out_flow.backward()
    out_torch = out_torch.sum()
    out_torch.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), input_torch.grad.numpy(), 1e-06, 1e-06)
    )


@flow.unittest.skip_unless_1n1d()
class TestConvTranspose(flow.unittest.TestCase):
    def test_ConvTranspose1d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_convtranspose1d_bias_false,
            _test_convtranspose1d_bias_true,
            _test_convtranspose1d_group_bias_false,
            _test_convtranspose1d_group_bias_true,
            _test_convtranspose1d_group_large_out_channel,
            _test_convtranspose1d_group_large_in_channel,
        ]
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
