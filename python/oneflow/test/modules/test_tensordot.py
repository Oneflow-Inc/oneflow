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
import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestTensordot(flow.unittest.TestCase):
    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_tensordot_intdim(test_case):
        device = random_device()
        dims = random()
        dims_list = [random().to(int).value() for i in range(dims.to(int).value() + 3)]
        x = random_tensor(
            ndim=3, dim0=dims_list[0], dim1=dims_list[1], dim2=dims_list[2],
        ).to(device)
        y = random_tensor(
            ndim=3,
            dim0=dims_list[0 + dims.to(int).value()],
            dim1=dims_list[1 + dims.to(int).value()],
            dim2=dims_list[2 + dims.to(int).value()],
        ).to(device)

        z = torch.tensordot(x, y, dims=3 - dims.to(int).value())
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_tensordot_list_dim(test_case):
        device = random_device()
        x = random_tensor(4, 1, 3, 2, 5).to(device)
        y = random_tensor(4, 4, 2, 3, 5).to(device)
        z = torch.tensordot(x, y, dims=[[1, 2, 0], [2, 1, 0]])
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-2)
    def test_tensordot_tuple_dim(test_case):
        device = random_device()
        x = random_tensor(4, 1, 3, 2, 5).to(device)
        y = random_tensor(4, 4, 2, 3, 5).to(device)
        z = torch.tensordot(x, y, dims=([1, 2, 0], [2, 1, 0]))
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_tensordot_list_neg_dim(test_case):
        device = random_device()
        x = random_tensor(4, 1, 3, 2, 5).to(device)
        y = random_tensor(4, 4, 2, 3, 5).to(device)
        z = torch.tensordot(x, y, dims=[[-3, -2, -4], [-2, -3, -4]])
        return z

    @autotest(check_graph=False, rtol=1e-2, atol=1e-3)
    def test_tensordot_backward(test_case):
        device = random_device()
        x = random_tensor(3, 3, 4, 5).to(device)
        y = random_tensor(2, 4, 5).to(device)
        z = torch.tensordot(x, y, dims=[[1, 2], [0, 1]])
        z.sum().backward()

    @autotest(check_graph=False)
    def test_tensordot_tensor_dim(test_case):
        def _test_tensor_dim(test_case, device):
            np_dim = np.array([[1, 2, 3], [1, 2, 3]], dtype=int)
            flow_dim = flow.tensor(np_dim).to(device)
            torch_dim = torch.tensor(np_dim).to(device)

            np_random_array = np.random.randn(2, 3, 4, 5)
            flow_tensor = flow.tensor(np_random_array).to(device)
            torch_tensor = torch.tensor(np_random_array).to(device)

            flow_result = flow.tensordot(flow_tensor, flow_tensor, dims=flow_dim)
            torch_result = torch.tensordot(torch_tensor, torch_tensor, dims=torch_dim)
            test_case.assertTrue(
                np.allclose(
                    flow_result.numpy(),
                    torch_result.cpu().numpy(),
                    rtol=0.0001,
                    atol=0.0001,
                )
            )

        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_tensor_dim(test_case, arg[0])

    @autotest(n=5, check_graph=False, rtol=1e-2, atol=1e-2)
    def test_tensordot_single_item_tensor_dim(test_case):
        device = random_device()
        dims = random_tensor(1, dim0=1, low=0, high=4, dtype=int).to(device)
        x = random_tensor(3, dim0=4, dim1=4, dim2=4).to(device)
        y = random_tensor(3, dim0=4, dim1=4, dim2=4).to(device)
        z = torch.tensordot(x, y, dims=dims)
        return z

    @autotest(n=5, rtol=1e-3, atol=1e-4)
    def test_tensordot_broadcast(test_case):
        device = random_device()
        x = random_tensor(4, 1, 1, 1, 1).to(device)
        y = random_tensor(4, 2, 3, 4, 5).to(device)
        z = torch.tensordot(x, y, dims=random(high=5).to(int).value())
        return z


if __name__ == "__main__":
    unittest.main()
