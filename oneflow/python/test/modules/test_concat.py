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

import torch
import oneflow.experimental as flow
from test_util import GenArgList
from automated_test_util import *


def _test_concat_origin(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )

    of_out = flow.cat([input1, input2], dim=0)
    np_out = np.concatenate((input1.numpy(), input2.numpy()), axis=0)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_concat_with_axis_one(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )

    of_out = flow.cat([input1, input2], dim=1)
    np_out = np.concatenate((input1.numpy(), input2.numpy()), axis=1)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_concat_with_three_tensor(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    input3 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )

    of_out = flow.cat([input1, input2, input3], dim=1)
    np_out = np.concatenate((input1.numpy(), input2.numpy(), input3.numpy()), axis=1)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_concat_with_three_tensor_backward(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input3 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )

    of_out = flow.cat([input1, input2, input3], dim=1)
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(input1.grad.numpy(), np.ones((2, 6, 5, 3)), 1e-4, 1e-4)
    )
    test_case.assertTrue(
        np.allclose(input2.grad.numpy(), np.ones((2, 6, 5, 3)), 1e-4, 1e-4)
    )
    test_case.assertTrue(
        np.allclose(input3.grad.numpy(), np.ones((2, 6, 5, 3)), 1e-4, 1e-4)
    )


def _test_concat_grad_and_no_grad(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=False,
    )

    of_out = flow.cat([input1, input2], dim=1)
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(input1.grad.numpy(), np.ones((2, 6, 5, 3)), 1e-4, 1e-4)
    )


def _test_concat_with_torch(test_case, device):
    ndim = np.random.randint(2, 5)
    n_tensors = np.random.randint(2, 5)
    cat_dim = np.random.randint(0, ndim - 1)

    shapes = []
    for i in range(ndim):
        shapes.append(np.random.randint(2, 5))

    flow_tensors = []
    torch_tensors = []
    for i in range(n_tensors):
        shapes[cat_dim] = np.random.randint(2, 5)
        np_arr = np.random.randn(*shapes)

        flow_input = flow.Tensor(
            np_arr, dtype=flow.float32, device=device, requires_grad=True
        )
        torch_input = torch.tensor(
            np_arr, dtype=torch.float32, device=device, requires_grad=True
        )

        flow_tensors.append(flow_input)
        torch_tensors.append(torch_input)

    of_out = flow.cat(flow_tensors, dim=cat_dim)
    of_out = of_out.sum()
    of_out.backward()

    torch_out = torch.cat(torch_tensors, dim=cat_dim)
    torch_out = torch_out.sum()
    torch_out.backward()

    test_case.assertTrue(
        np.allclose(torch_out.cpu().detach().numpy(), of_out.numpy(), 1e-4, 1e-4)
    )

    for i in range(n_tensors):
        test_case.assertTrue(
            np.allclose(
                torch_tensors[i].grad.cpu().detach().numpy(),
                flow_tensors[i].grad.numpy(),
                1e-4,
                1e-4,
            )
        )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestModule(flow.unittest.TestCase):
    def test_concat(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_concat_origin,
            _test_concat_with_axis_one,
            _test_concat_with_three_tensor,
            _test_concat_with_three_tensor_backward,
            _test_concat_grad_and_no_grad,
            _test_concat_with_torch,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
