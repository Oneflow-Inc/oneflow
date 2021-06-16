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

import oneflow.experimental as flow
from test_util import GenArgList


def _test_meshgrid_forawd(test_case, device):
    input1 = flow.Tensor(
        np.array([1, 2, 3]), dtype=flow.float32, device=flow.device(device),
    )
    input2 = flow.Tensor(
        np.array([4, 5, 6]), dtype=flow.float32, device=flow.device(device),
    )
    np_x, np_y = np.meshgrid(input1.numpy(), input2.numpy(), indexing="ij")
    of_x, of_y = flow.meshgrid(input1, input2)

    test_case.assertTrue(np.allclose(of_x.numpy(), np_x, 1e-4, 1e-4))
    test_case.assertTrue(np.allclose(of_y.numpy(), np_y, 1e-4, 1e-4))


def _test_meshgrid_forawd_scalr(test_case, device):
    input1 = flow.Tensor(np.array(1.0), dtype=flow.float32, device=flow.device(device),)
    input2 = flow.Tensor(np.array(2.0), dtype=flow.float32, device=flow.device(device),)
    np_x, np_y = np.meshgrid(input1.numpy(), input2.numpy(), indexing="ij")
    of_x, of_y = flow.meshgrid(input1, input2)

    test_case.assertTrue(np.allclose(of_x.numpy(), np_x, 1e-4, 1e-4))
    test_case.assertTrue(np.allclose(of_y.numpy(), np_y, 1e-4, 1e-4))


def _test_meshgrid_forawd_3tensor(test_case, device):
    input1 = flow.Tensor(
        np.array([1, 2, 3]), dtype=flow.float32, device=flow.device(device),
    )
    input2 = flow.Tensor(
        np.array([4, 5, 6]), dtype=flow.float32, device=flow.device(device),
    )
    input3 = flow.Tensor(
        np.array([7, 8, 9]), dtype=flow.float32, device=flow.device(device),
    )
    np_x, np_y, np_z = np.meshgrid(
        input1.numpy(), input2.numpy(), input3.numpy(), indexing="ij"
    )
    of_x, of_y, of_z = flow.meshgrid(input1, input2, input3)

    test_case.assertTrue(np.allclose(of_x.numpy(), np_x, 1e-4, 1e-4))
    test_case.assertTrue(np.allclose(of_y.numpy(), np_y, 1e-4, 1e-4))
    test_case.assertTrue(np.allclose(of_z.numpy(), np_z, 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMeshGrid(flow.unittest.TestCase):
    def test_meshgrid(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_meshgrid_forawd,
            _test_meshgrid_forawd_scalr,
            _test_meshgrid_forawd_3tensor,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
