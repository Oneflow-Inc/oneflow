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
from typing import Dict
import os


def _compare_offset_to_ndindex_with_np(
    input, dims, device_type, machine_ids, device_counts
):
    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    def np_offset_to_ndindex(input, dims):
        out = np.unravel_index(input, dims)
        return np.array(out).flatten()  # Flatten as 1-D array

    np_out_offset_to_ndindex = np_offset_to_ndindex(input, dims)

    # Convert Tuple to numpy.array for job_function
    of_dims = np.array(dims).astype(np.int32)

    @flow.global_function(
        type="predict", function_config=func_config,
    )
    def oneflow_offset_to_ndindex(
        of_offset: tp.Numpy.Placeholder(shape=input.shape, dtype=flow.int32),
        of_dims: tp.Numpy.Placeholder(shape=of_dims.shape, dtype=flow.int32),
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            of_offset_to_ndindex_out = flow.offset_to_ndindex(of_offset, of_dims)

        return of_offset_to_ndindex_out

    of_out_offset_to_ndindex = oneflow_offset_to_ndindex(input, of_dims)
    assert np.allclose(of_out_offset_to_ndindex, np_out_offset_to_ndindex)


def _gen_arg_dict(input, dims, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input"] = [input]
    arg_dict["dims"] = [dims]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestOffsetToNdIndex1n1d(flow.unittest.TestCase):
    def test_offset_to_ndindex_cpu(test_case):
        x = np.array([64]).astype(np.int32)
        dims = (8, 8, 8)
        arg_dict = _gen_arg_dict(
            input=x, dims=dims, device_type="cpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_offset_to_ndindex_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_offset_to_ndindex_gpu(test_case):
        x = np.array([13]).astype(np.int32)
        dims = (8, 8)
        arg_dict = _gen_arg_dict(
            input=x, dims=dims, device_type="gpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_offset_to_ndindex_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestOffsetToNdIndex1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_offset_to_ndindex_gpu_1n2d(test_case):
        x = np.array([25]).astype(np.int32)
        dims = (8, 8)
        arg_dict = _gen_arg_dict(
            input=x, dims=dims, device_type="gpu", machine_ids="0:0-1", device_counts=2
        )
        for arg in GenArgList(arg_dict):
            _compare_offset_to_ndindex_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
