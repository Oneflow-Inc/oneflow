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

import oneflow.experimental as flow
from test_util import GenArgList


def _test_argwhere(test_case, device):
    np_input = np.random.randn(2, 4, 5, 6)
    input = flow.Tensor(np_input, device=flow.device(device))
    of_out = flow.argwhere(input)
    np_list = []
    for i in range(2):
        for j in range(4):
            for k in range(5):
                for l in range(6):
                    if np_input[i][j][k][k]:
                        np_list.append([i, j, k, l])
    np_out = np.array(np_list)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))

def _test_argwhere_grad(test_case, device):
    np_input = [[1.0, 2.0], [0.0, 0.2]]
    input = flow.Tensor(np_input, device=flow.device(device), requires_grad=True)
    of_out = flow.argwhere(input)
    of_out = of_out.sum()
    of_out.backward()
    print(input.grad.numpy().tolist())


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestArgwhere(flow.unittest.TestCase):
    def test_argwhere(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_argwhere, _test_argwhere_grad]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
