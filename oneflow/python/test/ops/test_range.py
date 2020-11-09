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
from typing import List


def compare_range_with_np_CPU(device_type, machine_ids, device_counts):
    assert device_type in ["cpu"]

    flow.clear_default_session()
    flow.env.init()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    @flow.global_function(function_config=func_config)
    def oneflow_range() -> List[tp.Numpy]:
        with flow.scope.placement(device_type, machine_ids):
            out_1 = flow.range(2, 10, 3, dtype=flow.float64, name="range_float64")
            out_2 = flow.range(0, 10, 1, dtype=flow.float32, name="range_float32")
            out_3 = flow.range(10, dtype=flow.int32, name="range_int32")
            out_4 = flow.range(0, 10, 2, dtype=flow.int64, name="range_int64")

        return [out_1, out_2, out_3, out_4]

    def np_range():
        np_out_1 = np.arange(2, 10, 3).astype(np.float64)
        np_out_2 = np.arange(0, 10, 1).astype(np.float32)
        np_out_3 = np.arange(10).astype(np.int32)
        np_out_4 = np.arange(0, 10, 2).astype(np.int64)

        return [np_out_1, np_out_2, np_out_3, np_out_4]

    of_out_list = oneflow_range()
    np_out_list = np_range()

    for i in range(len(of_out_list)):
        assert np.array_equal(of_out_list[i], np_out_list[i])


def compare_range_with_np_GPU(device_type, machine_ids, device_counts):
    assert device_type in ["gpu"]

    flow.clear_default_session()
    flow.env.init()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    @flow.global_function(function_config=func_config)
    def oneflow_range_gpu() -> List[tp.Numpy]:
        with flow.scope.placement(device_type, machine_ids):
            out_1 = flow.range(2, 10, 3, dtype=flow.float64, name="range_float64")
            out_2 = flow.range(0, 10, 1, dtype=flow.float32, name="range_float32")
            out_3 = flow.range(0, 10, 1, dtype=flow.float32, name="range_float16")
            # Oneflow doesn't support float16 output, so we need to cast it to float32
            out_3 = flow.cast(out_3, dtype=flow.float32)
            out_4 = flow.range(10, dtype=flow.int32, name="range_int32")
            out_5 = flow.range(0, 10, 2, dtype=flow.int64, name="range_int64")

        return [out_1, out_2, out_3, out_4, out_5]

    def np_range_gpu():
        np_out_1 = np.arange(2, 10, 3).astype(np.float64)
        np_out_2 = np.arange(0, 10, 1).astype(np.float32)
        np_out_3 = np.arange(0, 10, 1).astype(np.float16)
        np_out_4 = np.arange(10).astype(np.int32)
        np_out_5 = np.arange(0, 10, 2).astype(np.int64)

        return [np_out_1, np_out_2, np_out_3, np_out_4, np_out_5]

    of_out_list = oneflow_range_gpu()
    np_out_list = np_range_gpu()

    for i in range(len(of_out_list)):
        assert np.array_equal(of_out_list[i], np_out_list[i])


@flow.unittest.skip_unless_1n1d()
class Testrange1n1d(flow.unittest.TestCase):
    def test_range_cpu(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        for arg in GenArgList(arg_dict):
            compare_range_with_np_CPU(*arg)

    def test_range_gpu(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        for arg in GenArgList(arg_dict):
            compare_range_with_np_GPU(*arg)


@flow.unittest.skip_unless_1n2d()
class Testrange1n2d(flow.unittest.TestCase):
    def test_range_gpu_1n2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["device_counts"] = [2]
        for arg in GenArgList(arg_dict):
            compare_range_with_np_GPU(*arg)


if __name__ == "__main__":
    unittest.main()
