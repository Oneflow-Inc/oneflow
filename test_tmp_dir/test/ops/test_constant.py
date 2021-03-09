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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type
import math

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def _test(test_case, device_type, type_name_value):
    type_name, value = type_name_value
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    flow_type = type_name_to_flow_type[type_name]
    np_type = type_name_to_np_type[type_name]
    shape = (1024, 1024)

    @flow.global_function(function_config=func_config)
    def constant_job():
        with flow.scope.placement(device_type, "0:0"):
            return flow.constant(value, dtype=flow_type, shape=shape)

    of_out = constant_job().get().numpy()
    test_case.assertTrue(np.array_equal(of_out, np.full(shape, value).astype(np_type)))


@flow.unittest.skip_unless_1n1d()
class TestConstant(flow.unittest.TestCase):
    def test_constant(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["type_name_value"] = [
            ("float32", 0),
            ("float32", 0.0),
            ("float32", 1),
            ("float32", 1.0),
            ("float32", -1),
            ("float32", -1.0),
            ("float32", math.pi),
            ("float32", -math.pi),
            ("float32", float("inf")),
            ("float32", float("-inf")),
            ("int32", 0),
            ("int32", 0.0),
            ("int32", 1),
            ("int32", 1.0),
            ("int32", -1),
            ("int32", -1.0),
            ("int32", 2 ** 31 - 1),
            ("int32", -(2 ** 31)),
        ]
        for arg in GenArgList(arg_dict):
            _test(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
