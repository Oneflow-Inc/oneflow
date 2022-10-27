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
from oneflow.test_utils.automated_test_util import *
import oneflow.unittest


class DataChecker:
    check_list = [
        "mean",
        "std",
        "min",
        "max",
        "value",
        "lambda",
    ]

    def __init__(self, **kwargs):
        self.checkers = {}
        for key in self.check_list:
            if key in kwargs:
                self.checkers[key] = kwargs[key]

    def __call__(self, test_case, tensor):
        for func in ["mean", "std", "min", "max"]:
            if func in self.checkers:
                of_res = eval(f"tensor.{func}")().numpy()
                checker_res = self.checkers[func]
                test_case.assertTrue(
                    np.allclose(of_res, checker_res, rtol=1e-2, atol=1e-2),
                    f"{func} not equal, {of_res} vs {checker_res}",
                )

        if "value" in self.checkers:
            test_case.assertTrue(np.all(tensor.numpy() == self.checkers["value"]))

        if "lambda" in self.checkers:
            test_case.assertTrue(
                np.allclose(
                    tensor.numpy(),
                    self.checkers["lambda"](tensor.shape),
                    rtol=1e-4,
                    atol=1e-4,
                )
            )


# NOTE(wyg): register initializers to this list
check_func_list = [
    # oneflow.nn.init.normal_
    {
        "func": flow.nn.init.normal_,
        "params": {"mean": 0.0, "std": 1.0},
        "checker": DataChecker(mean=0.0, std=1.0),
    },
    # oneflow.nn.init.xavier_normal_
    {
        "func": flow.nn.init.xavier_normal_,
        "params": {"gain": 1.0},
        "checker": DataChecker(mean=0.0, std=0.0625),
    },
    # oneflow.nn.init.kaiming_normal_
    {
        "func": flow.nn.init.kaiming_normal_,
        "params": {"mode": "fan_in"},
        "checker": DataChecker(mean=0.0, std=0.0883883476),
    },
    {
        "func": flow.nn.init.kaiming_normal_,
        "params": {"mode": "fan_out"},
        "checker": DataChecker(mean=0.0, std=0.0883883476),
    },
    {
        "func": flow.nn.init.kaiming_normal_,
        "params": {"mode": "fan_in", "a": 2.0, "nonlinearity": "leaky_relu"},
        "checker": DataChecker(mean=0.0, std=0.0395284708),
    },
    {
        "func": flow.nn.init.kaiming_normal_,
        "params": {"mode": "fan_in", "a": 2.0, "nonlinearity": "linear"},
        "checker": DataChecker(mean=0.0, std=0.0625),
    },
    # TODO: test more initializer
]


@oneflow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestInitializer(flow.unittest.TestCase):
    def test_initializer(test_case):
        default_shape = (256, 256)
        for device in ["cpu", "cuda"]:
            for check_func in check_func_list:
                tensor = flow.empty(*default_shape, device=flow.device(device))
                check_func["func"](tensor, **check_func["params"])
                try:
                    check_func["checker"](test_case, tensor)
                except AssertionError as e:
                    print(
                        f"Failed: {check_func['func'].__name__} {check_func['params']}"
                    )
                    raise e


if __name__ == "__main__":
    unittest.main()
