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


def _test_norm_vector(test_case, device):
    input = flow.Tensor(
        np.random.randn(10,), dtype=flow.float32, device=flow.device(device)
    )
    of_out_1 = flow.norm(input)
    of_out_2 = flow.norm(input, ord = 0)
    of_out_3 = flow.norm(input, ord = 3)
    np_out_1 = np.linalg.norm(input.numpy())
    np_out_2 = np.linalg.norm(input.numpy(), ord = 0)
    np_out_3 = np.linalg.norm(input.numpy(), ord = 3)
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out_1, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out_2, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_3.numpy(), np_out_3, 1e-5, 1e-5))


def _test_norm_matrix(test_case, device):
    input = flow.Tensor(
        np.random.randn(5,4), dtype=flow.float32, device=flow.device(device)
    )
    of_out_1 = flow.norm(input)
    of_out_2 = flow.norm(input, dim = 0)
    of_out_3 = flow.norm(input, dim = 1, keepdim = True)
    np_out_1 = np.linalg.norm(input.numpy())
    np_out_2 = np.linalg.norm(input.numpy(), axis = 0)
    np_out_3 = np.linalg.norm(input.numpy(), axis = 1, keepdims = True)
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out_1, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out_2, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_3.numpy(), np_out_3, 1e-5, 1e-5))

class TestNormModule(flow.unittest.TestCase):
    def test_norm(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_norm_vector,
            _test_norm_matrix
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
