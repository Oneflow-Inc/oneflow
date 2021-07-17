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

import oneflow as flow
import numpy as np
import oneflow.typing as oft
from test_util import GenArgDict


def _test_slice_assign(
    test_case,
    var_shape,
    slice_tuples,
    split_axis,
    dst_device_tag,
    flow_dtype,
    device_num,
):
    flow.clear_default_session()
    value_shape = [(s[1] - s[0] - 1) // s[2] + 1 for s in slice_tuples]
    flow_to_np_dtype_dict = {
        flow.int8: np.int8,
        flow.float: np.single,
    }
    np_dtype = flow_to_np_dtype_dict[flow_dtype]
    value = np.random.uniform(low=-10, high=10, size=value_shape).astype(np_dtype)
    if dst_device_tag == "gpu":
        flow.config.gpu_device_num(device_num)

    def get_var():
        return flow.get_variable(
            name="var",
            shape=var_shape,
            dtype=flow_dtype,
            initializer=flow.constant_initializer(0, dtype=flow_dtype),
            distribute=flow.distribute.split(split_axis),
        )

    @flow.global_function()
    def assign_fn(value_def: oft.Numpy.Placeholder(value.shape, dtype=flow_dtype)):
        with flow.scope.placement(dst_device_tag, "0:0-{}".format(device_num - 1)):
            var = get_var()
            flow.experimental.logical_slice_assign(var, value_def, slice_tuples)

    @flow.global_function()
    def identity_fn():
        with flow.scope.placement(dst_device_tag, "0:0-{}".format(device_num - 1)):
            var = get_var()
            return flow.identity(var)

    assign_fn(value)
    of_res = identity_fn().get().numpy()

    np_res = np.zeros(var_shape).astype(np_dtype)
    slice_objs = []
    for s in slice_tuples:
        slice_objs.append(slice(s[0], s[1], s[2]))
    np_res[tuple(slice_objs)] = value

    test_case.assertTrue(np.array_equal(of_res, np_res))


@flow.unittest.skip_unless_1n4d()
class TestSliceAssign(flow.unittest.TestCase):
    def test_slice_assign_4dim_4d(test_case):
        var_shape = (30, 40, 20, 15)
        slice_tuples = [(10, 20, 3), (1, 30, 4), (3, 16, 2), (5, 11, 1)]
        arg_dict = OrderedDict()
        arg_dict["split_axis"] = list(range(4))
        arg_dict["dst_device_tag"] = ["cpu", "gpu"]
        arg_dict["flow_dtype"] = [flow.float, flow.int8]
        arg_dict["device_num"] = [4]
        for arg in GenArgDict(arg_dict):
            _test_slice_assign(test_case, var_shape, slice_tuples, **arg)

    def test_slice_assign_negative_start_stop_4dim_4d(test_case):
        var_shape = (30, 40, 20, 15)
        slice_tuples = [(10, 20, 3), (-39, -10, 4), (-15, -5, 2), (5, 11, 1)]
        arg_dict = OrderedDict()
        arg_dict["split_axis"] = list(range(4))
        arg_dict["dst_device_tag"] = ["cpu", "gpu"]
        arg_dict["flow_dtype"] = [flow.float]
        arg_dict["device_num"] = [4]
        for arg in GenArgDict(arg_dict):
            _test_slice_assign(test_case, var_shape, slice_tuples, **arg)

    def test_slice_assign_2dim_3d(test_case):
        var_shape = (30, 40)
        slice_tuples = [(10, 20, 3), (1, 30, 4)]
        arg_dict = OrderedDict()
        arg_dict["split_axis"] = list(range(2))
        arg_dict["dst_device_tag"] = ["cpu", "gpu"]
        arg_dict["flow_dtype"] = [flow.float]
        arg_dict["device_num"] = [3]
        for arg in GenArgDict(arg_dict):
            _test_slice_assign(test_case, var_shape, slice_tuples, **arg)
