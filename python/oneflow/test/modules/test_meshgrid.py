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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_meshgrid_forawd(test_case, device, indexing):
    input1 = flow.tensor(
        np.array([1, 2, 3]), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.tensor(
        np.array([4, 5, 6]), dtype=flow.float32, device=flow.device(device)
    )
    (np_x, np_y) = np.meshgrid(input1.numpy(), input2.numpy(), indexing=indexing)
    (of_x, of_y) = flow.meshgrid(input1, input2, indexing=indexing)
    test_case.assertTrue(np.allclose(of_x.numpy(), np_x, 0.0001, 0.0001))


def _test_meshgrid_forawd_scalar(test_case, device, indexing):
    input1 = flow.tensor(np.array(1.0), dtype=flow.float32, device=flow.device(device))
    input2 = flow.tensor(np.array(2.0), dtype=flow.float32, device=flow.device(device))
    (np_x, np_y) = np.meshgrid(input1.numpy(), input2.numpy(), indexing=indexing)
    (of_x, of_y) = flow.meshgrid(input1, input2, indexing=indexing)
    test_case.assertTrue(np.allclose(of_x.numpy(), np_x, 0.0001, 0.0001))


def _test_meshgrid_forawd_3tensor(test_case, device, indexing):
    input1 = flow.tensor(
        np.array([1, 2, 3]), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.tensor(
        np.array([4, 5, 6]), dtype=flow.float32, device=flow.device(device)
    )
    input3 = flow.tensor(
        np.array([7, 8, 9]), dtype=flow.float32, device=flow.device(device)
    )
    (np_x, np_y, np_z) = np.meshgrid(
        input1.numpy(), input2.numpy(), input3.numpy(), indexing=indexing
    )
    (of_x, of_y, of_z) = flow.meshgrid(input1, input2, input3, indexing=indexing)
    test_case.assertTrue(np.allclose(of_x.numpy(), np_x, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestMeshGridModule(flow.unittest.TestCase):
    def test_meshgrid(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_meshgrid_forawd,
            _test_meshgrid_forawd_scalar,
            _test_meshgrid_forawd_3tensor,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["indexing"] = ["ij", "xy"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(auto_backward=False, check_graph=True)
    @unittest.skip("pytorch 1.9.0 exist not indexing")
    def test_meshgrid_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=3, requires_grad=False).to(device)
        y = random_tensor(ndim=1, dim0=3, requires_grad=False).to(device)
        res = torch.meshgrid(x, y)
        return res[0], res[1]

    @autotest(auto_backward=False)
    def test_meshgrid_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=0).to(device)
        res = torch.meshgrid(x, y)

    @autotest(auto_backward=True)
    @unittest.skip("pytorch 1.9.0 exist not indexing")
    def test_meshgrid_with_random_data_xy(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=random(1, 6)).to(device)
        y = random_tensor(ndim=1, dim0=random(1, 6)).to(device)
        res = torch.meshgrid(x, y, indexing="xy")
        return torch.cat((res[0], res[1]), 0)

    @autotest(auto_backward=True)
    @unittest.skip("pytorch 1.9.0 exist not indexing")
    def test_meshgrid_with_random_data_size(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=random(1, 6)).to(device)
        res = torch.meshgrid(x, indexing="xy")
        return res[0]

    @autotest(n=3)
    def test_meshgrid_tuple_list_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=random(1, 6)).to(device)
        y = random_tensor(ndim=1, dim0=random(1, 6)).to(device)
        res1 = torch.meshgrid((x, y))
        res2 = torch.meshgrid([x, y])
        return torch.cat((res1[0], res1[1], res2[0], res2[1]), 0)


if __name__ == "__main__":
    unittest.main()
