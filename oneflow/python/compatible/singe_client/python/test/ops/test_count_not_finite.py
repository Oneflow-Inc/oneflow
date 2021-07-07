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
import oneflow as flow
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft


def _run_count_test(test_case, device_type, x_shape, dtype):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def count_not_finite_job(
        x: oft.Numpy.Placeholder(x_shape, dtype=type_name_to_flow_type[dtype]),
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.count_not_finite(x)

    x = np.random.randn(*x_shape).astype(type_name_to_np_type[dtype])
    x[0] = np.nan
    x[5][4] = np.inf
    y = count_not_finite_job(x).get()
    np_y = x.size - np.sum(np.isfinite(x))
    assert y.numpy() == np_y


def _run_multi_count_test(
    test_case, device_type, x1_shape, x2_shape, dtype, x1_count, x2_count,
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def multi_count_not_finite_job(
        x1: oft.Numpy.Placeholder(x1_shape, dtype=type_name_to_flow_type[dtype]),
        x2: oft.Numpy.Placeholder(x2_shape, dtype=type_name_to_flow_type[dtype]),
    ):
        x_list = []
        for i in range(x1_count):
            x_list.append(x1)
        for i in range(x2_count):
            x_list.append(x2)
        with flow.scope.placement(device_type, "0:0"):
            return flow.multi_count_not_finite(x_list)

    x1 = np.random.randn(*x1_shape).astype(type_name_to_np_type[dtype])
    x1[0] = np.nan
    x1[3] = np.inf
    x2 = np.random.randn(*x2_shape).astype(type_name_to_np_type[dtype])
    x2[2] = np.inf
    x2[6, 5] = np.nan
    y = multi_count_not_finite_job(x1, x2).get()
    x1_not_finite = x1.size - np.sum(np.isfinite(x1))
    x2_not_finite = x2.size - np.sum(np.isfinite(x2))
    np_y = x1_not_finite * x1_count + x2_not_finite * x2_count
    assert y.numpy() == np_y


@flow.unittest.skip_unless_1n1d()
class TestCountNotFinite(flow.unittest.TestCase):
    def test_count_not_finite(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["x_shape"] = [(10, 30)]
        arg_dict["dtype"] = ["float32", "double"]
        for arg in GenArgList(arg_dict):
            _run_count_test(*arg)

    def test_multi_count_not_finite(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["x1_shape"] = [(10, 20, 20)]
        arg_dict["x2_shape"] = [(10, 20)]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["x1_count"] = [10]
        arg_dict["x2_count"] = [30]
        for arg in GenArgList(arg_dict):
            _run_multi_count_test(*arg)


if __name__ == "__main__":
    unittest.main()
