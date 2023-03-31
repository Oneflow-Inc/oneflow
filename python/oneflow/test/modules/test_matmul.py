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
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    @autotest(check_graph=True, rtol=1e-2, atol=1e-3)
    def test_flow_matmul_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        z = torch.matmul(x, y)
        return z

    @autotest(check_graph=True, rtol=1e-2, atol=1e-4)
    def test_flow_tensor_matmul_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        return x.matmul(y)

    @autotest(n=5, check_graph=False)
    def test_flow_tensor_matmul_with_random_int_data(test_case):
        x = np.random.randint(10, 21, size=5)
        y = np.random.randint(1, 14, size=(5, 4))
        torch_x = torch.from_numpy(x).to(torch.int)
        torch_y = torch.from_numpy(y).to(torch.int)
        torch_output_numpy = torch_x.matmul(torch_y).numpy()
        flow_x = flow.tensor(x).to(flow.int)
        flow_y = flow.tensor(y).to(flow.int)
        flow_output_numpy = flow_x.matmul(flow_y).numpy()
        test_case.assertTrue(
            np.allclose(flow_output_numpy, torch_output_numpy, 1e-05, 1e-05)
        )

    @autotest(n=5, check_graph=True, rtol=1e-2, atol=1e-3)
    def test_flow_tensor_broadcast_matmul_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=4, dim3=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        return x.matmul(y)

    @autotest(n=5, check_graph=True, rtol=1e-2, atol=1e-3)
    def test_flow_tensor_x_broadcast_y_matmul(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=4, dim2=k).to(device)
        return x.matmul(y)

    @autotest(n=5, check_graph=True, rtol=1e-2, atol=1e-4)
    def test_flow_tensor_broadcast_matmul_with_same_dims(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=4, dim1=1, dim3=k).to(device)
        y = random_tensor(ndim=4, dim0=1, dim2=k).to(device)
        return x.matmul(y)

    @autotest(check_graph=True, rtol=1e-2, atol=1e-3)
    def test_flow_mm_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        z = torch.mm(x, y)
        return z

    @autotest(n=5, check_graph=True)
    def test_flow_mv_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=1, dim0=k).to(device)
        z = torch.mv(x, y)
        return z

    @profile(torch.mv)
    def profile_mv(test_case):
        torch.mv(torch.ones(32, 64), torch.ones(64))

    @autotest(n=5, check_graph=True, rtol=1e-2, atol=1e-4)
    def test_flow_vector_matrix_product_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=1, dim0=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        z = torch.matmul(x, y)
        return z


if __name__ == "__main__":
    unittest.main()
