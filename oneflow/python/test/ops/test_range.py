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
import oneflow as flow
import numpy as np
import oneflow.typing as tp
from test_util import GenArgList
import unittest
from collections import OrderedDict


def compare_range_with_np(
    device_type, datatype, machine_ids, device_counts, start, limit, delta
):
    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    flow.env.init()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_data_type(datatype[0])

    @flow.global_function(function_config=func_config)
    def oneflow_range() -> tp.Numpy:
        with flow.scope.placement(device_type, machine_ids):
            return flow.range(start, limit, delta, dtype=datatype[0])

    of_out = oneflow_range()
    np_out = np.arange(start, limit, delta, dtype=datatype[1])
    assert np.array_equal(of_out, np_out)


@flow.unittest.skip_unless_1n1d()
class Testrange1n1d(flow.unittest.TestCase):
    def test_range1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["datatype"] = [[flow.float32, np.float32]]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        arg_dict["start"] = [0]
        arg_dict["limit"] = [10]
        arg_dict["delta"] = [1]
        # test like flow.range(0, 10, 1)
        for arg in GenArgList(arg_dict):
            compare_range_with_np(*arg)

    def test_range2(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["datatype"] = [[flow.int32, np.int32]]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        arg_dict["start"] = [10]
        arg_dict["limit"] = [None]
        arg_dict["delta"] = [1]
        # test like flow.range(10)
        for arg in GenArgList(arg_dict):
            compare_range_with_np(*arg)

    def test_range3(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["datatype"] = [[flow.int64, np.int64]]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        arg_dict["start"] = [0]
        arg_dict["limit"] = [10]
        arg_dict["delta"] = [2]
        # test like flow.range(0, 10, 2) -> [0, 2, 4, 6, 8]
        for arg in GenArgList(arg_dict):
            compare_range_with_np(*arg)

    def test_range4(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["datatype"] = [[flow.float32, np.float32]]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        arg_dict["start"] = [1]
        arg_dict["limit"] = [10]
        arg_dict["delta"] = [3]
        # test like flow.range(1, 10, 3) -> [1, 4, 7]
        for arg in GenArgList(arg_dict):
            compare_range_with_np(*arg)

    def test_range5(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["datatype"] = [[flow.float64, np.float64]]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        arg_dict["start"] = [2]
        arg_dict["limit"] = [10]
        arg_dict["delta"] = [3]
        # test like flow.range(2, 10, 3) -> [2, 5, 8]
        for arg in GenArgList(arg_dict):
            compare_range_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testrange1n2d(flow.unittest.TestCase):
    def test_range1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["datatype"] = [[flow.int32, np.int32]]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["device_counts"] = [2]
        arg_dict["start"] = [10]
        arg_dict["limit"] = [None]
        arg_dict["delta"] = [1]
        # test like flow.range(10)
        for arg in GenArgList(arg_dict):
            compare_range_with_np(*arg)

    def test_range2(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["datatype"] = [[flow.int64, np.int64]]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["device_counts"] = [2]
        arg_dict["start"] = [0]
        arg_dict["limit"] = [10]
        arg_dict["delta"] = [2]
        # test like flow.range(0, 10, 2) -> [0, 2, 4, 6, 8]
        for arg in GenArgList(arg_dict):
            compare_range_with_np(*arg)

    def test_range3(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["datatype"] = [[flow.float32, np.float32]]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["device_counts"] = [2]
        arg_dict["start"] = [1]
        arg_dict["limit"] = [10]
        arg_dict["delta"] = [3]
        # test like flow.range(1, 10, 3) -> [1, 4, 7]
        for arg in GenArgList(arg_dict):
            compare_range_with_np(*arg)

    def test_range4(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["datatype"] = [[flow.float64, np.float64]]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["device_counts"] = [2]
        arg_dict["start"] = [2]
        arg_dict["limit"] = [10]
        arg_dict["delta"] = [3]
        # test like flow.range(2, 10, 3) -> [2, 5, 8]
        for arg in GenArgList(arg_dict):
            compare_range_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
