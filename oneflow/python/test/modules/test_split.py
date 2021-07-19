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
import oneflow.experimental as flow
from test_util import GenArgList
import numpy as np
import unittest
from collections import OrderedDict
import random


def _test_split_sections(test_case, device):
    x = np.random.rand(4,2,3,4)
    x_tensor = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    out_np = np.split(x, 2)
    out_of = flow.split(x_tensor, 2).numpy()
    test_case.assertTrue(np.allclose(out_np, out_of, 1e-5, 1e-5))

def _test_split_sizes(test_case, device):
    x = np.random.rand(4,2,3,4)
    x_tensor = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    out_np = np.split(x, [1, 3])
    out_of = flow.split(x_tensor, [1, 3]).numpy()
    test_case.assertTrue(np.allclose(out_np, out_of, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestStack(flow.unittest.TestCase):
    def test_stack(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            # _test_split,
            _test_split_sizes
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
