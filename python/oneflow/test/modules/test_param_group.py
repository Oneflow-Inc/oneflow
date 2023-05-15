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


@flow.unittest.skip_unless_1n1d()
class TestParamGroup(flow.unittest.TestCase):
    def test_ParamGroup(test_case):
        parameters = {
            "params": [flow.ones(10), flow.ones(5)],
            "lr": 0.01,
        }
        default_options = {
            "test_float": 1e-3,
            "test_int": 6,
            "test_list": [1, 2, 3],
            "test_tensor": flow.ones(10),
            "test_str": "test",
        }

        pg = flow.optim.optimizer.ParamGroup(parameters, default_options)

        test_case.assertEqual(pg["test_float"], 1e-3)
        test_case.assertEqual(pg["test_int"], 6)
        test_case.assertTrue(np.array_equal(pg.get("test_list"), [1, 2, 3]))
        test_case.assertTrue(
            np.array_equal(pg.get("test_tensor").numpy(), flow.ones(10).numpy())
        )
        test_case.assertEqual(pg["test_str"], "test")
        test_case.assertTrue("params" in pg.keys())
        test_case.assertTrue(
            np.array_equal(pg["params"][0].numpy(), flow.ones(10).numpy())
        )
        test_case.assertTrue(
            np.array_equal(pg["params"][1].numpy(), flow.ones(5).numpy())
        )
        test_case.assertEqual(pg["lr"], 0.01)


if __name__ == "__main__":
    unittest.main()
