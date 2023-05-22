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

import oneflow as flow

from oneflow.test_utils.test_util import GenArgDict
import numpy as np
import torch


test_compute_op_list = [
    "+",
    "-",
    "*",
    "/",
    "**",
    "//",
    "%",
]

test_login_op_list = [
    "^",
    "&",
    "|",
]

test_compare_op_list = [
    "==",
    "!=",
]


def _test_compute_operator(test_case, shape, dtype):
    random_tensor = np.random.randn(*shape).astype(dtype)
    x_flow = flow.tensor(random_tensor)
    x_torch = torch.tensor(random_tensor)
    random_numpy = np.random.randn(*shape)

    for op in test_compute_op_list:
        if op in ["**", "//", "%"]:
            random_tensor = np.random.randint(1, 100, size=shape)
            random_numpy = np.random.randint(1, 10, size=shape)
        else:
            random_tensor = np.random.randn(*shape)
            random_numpy = np.random.randn(*shape)

        x_flow = flow.tensor(random_tensor)
        x_torch = torch.tensor(random_tensor)

        z_flow = eval(f"x_flow {op} random_numpy")
        z_torch = eval(f"x_torch {op} random_numpy")
        test_case.assertTrue(np.allclose(z_flow.numpy(), z_torch.numpy()))

        # TODO:support for "+=" compatibility
        if op not in ["**", "+"]:
            exec(f"x_flow {op}= random_numpy")
            exec(f"x_torch {op}= random_numpy")
            test_case.assertTrue(
                np.allclose(z_flow.numpy(), z_torch.numpy(), 1e-05, 1e-05)
            )


def _test_logic_operator(test_case, shape):
    random_tensor = np.random.randint(100, size=shape)
    x_flow = flow.tensor(random_tensor, dtype=flow.int64)
    x_torch = torch.tensor(random_tensor, dtype=torch.int64)
    random_numpy = np.random.randint(100, size=shape)

    for op in test_login_op_list:
        z_flow = eval(f"x_flow {op} random_numpy")
        z_torch = eval(f"x_torch {op} random_numpy")
        test_case.assertTrue(np.allclose(z_flow.numpy(), z_torch.numpy(), 1e-05, 1e-05))


def _test_compare_operator(test_case, shape):
    random_tensor = np.random.randint(100, size=shape)
    x_flow = flow.tensor(random_tensor, dtype=flow.int64)
    x_torch = torch.tensor(random_tensor, dtype=torch.int64)
    random_numpy = np.random.randint(100, size=shape)

    for op in test_compare_op_list:
        flow_bool_value = eval(f"x_flow {op} random_numpy")
        torch_bool_value = eval(f"x_torch {op} random_numpy")
        print(flow_bool_value)
        print(torch_bool_value)
        test_case.assertTrue(flow_bool_value, torch_bool_value)


@flow.unittest.skip_unless_1n1d()
class TestTensorAndNdarrayCompatibility(flow.unittest.TestCase):
    def test_op_compatibility(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["dtype"] = [np.float32, np.float64]

        for arg in GenArgDict(arg_dict):
            _test_compute_operator(test_case, **arg)
            # TODO(yzm):support compare  operator Compatibility
            # _test_compare_operator(test_case, **arg)

            # TODO(yzm):fix the logic op bug
            # _test_logic_operator(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
