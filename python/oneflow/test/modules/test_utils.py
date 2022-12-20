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
import torch
from torch._utils import _flatten_dense_tensors as torch_flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors as torch_unflatten_dense_tensors
from oneflow._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgList


def _test_flatten_dense_tensors(test_case, device):
    torch_x = torch.randn(6, 6, device=device)
    x = flow.utils.tensor.from_torch(torch_x)
    torch_x_flatten = torch_flatten_dense_tensors([torch_x])
    x_flatten = _flatten_dense_tensors([x])
    test_case.assertTrue(np.array_equal(torch_x_flatten.size(), x_flatten.size()))
    torch_x_flatten = torch_flatten_dense_tensors([torch_x, torch_x, torch_x])
    x_flatten = _flatten_dense_tensors([x, x, x])
    test_case.assertTrue(np.array_equal(torch_x_flatten.size(), x_flatten.size()))
    test_case.assertTrue(
        np.allclose(
            torch_x_flatten.cpu().numpy(), x_flatten.cpu().numpy(), 1e-05, 1e-05
        )
    )


def _test_unflatten_dense_tensors(test_case, device):
    torch_flat = torch.randn(6, 1, device=device)
    torch_x1 = torch.randn(2, 1, device=device)
    torch_x2 = torch.randn(2, 1, device=device)
    torch_x3 = torch.randn(2, 1, device=device)
    torch_tensors = [
        torch_x1,
        torch_x2,
        torch_x3,
    ]
    tensors = [
        flow.utils.tensor.from_torch(torch_x1),
        flow.utils.tensor.from_torch(torch_x2),
        flow.utils.tensor.from_torch(torch_x3),
    ]
    torch_outputs = torch_unflatten_dense_tensors(torch_flat, torch_tensors)
    outputs = _unflatten_dense_tensors(
        flow.utils.tensor.from_torch(torch_flat), tensors
    )
    for i in range(len(outputs)):
        test_case.assertTrue(np.array_equal(torch_outputs[i].size(), outputs[i].size()))
        test_case.assertTrue(
            np.allclose(
                torch_outputs[i].cpu().numpy(), outputs[i].cpu().numpy(), 1e-05, 1e-05
            )
        )


@flow.unittest.skip_unless_1n1d()
class TestUtilsFunction(flow.unittest.TestCase):
    def test_utils_function(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_flatten_dense_tensors,
            _test_unflatten_dense_tensors,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
