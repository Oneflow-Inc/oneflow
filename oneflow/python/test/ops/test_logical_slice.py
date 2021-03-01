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
import test_global_storage
from test_util import GenArgDict


def _test_logical_slice(
    test_case, var_shape, slice_tuples, split_axis, device_tag, flow_dtype, device_num
):
    flow.clear_default_session()
    if device_tag == "gpu":
        flow.config.gpu_device_num(device_num)

    @flow.global_function()
    def slice_fn():
        with flow.scope.placement(device_tag, "0:0-{}".format(device_num - 1)):
            var = flow.get_variable(
                name="var",
                shape=var_shape,
                dtype=flow_dtype,
                initializer=flow.random_uniform_initializer(-10, 10, dtype=flow_dtype),
                distribute=flow.distribute.split(split_axis),
            )
            flow.watch(var, test_global_storage.Setter("var"))
            ret = flow.experimental.logical_slice(var, slice_tuples)
            return ret

    of_res = slice_fn().get().numpy()

    var_np = test_global_storage.Get("var")
    slice_objs = []
    for s in slice_tuples:
        slice_objs.append(slice(s[0], s[1], s[2]))
    test_case.assertTrue(np.array_equal(of_res, var_np[tuple(slice_objs)]))


class TestLogicalSlice(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_logical_slice_4dim_2d(test_case):
        var_shape = (30, 40, 20, 15)
        slice_tuples = [(10, 20, 3), (1, 30, 4), (3, 16, 2), (5, 11, 1)]
        arg_dict = OrderedDict()
        arg_dict["split_axis"] = list(range(4))
        arg_dict["device_tag"] = ["cpu", "gpu"]
        arg_dict["flow_dtype"] = [flow.float, flow.int8]
        arg_dict["device_num"] = [2]
        for arg in GenArgDict(arg_dict):
            _test_logical_slice(test_case, var_shape, slice_tuples, **arg)

    @flow.unittest.skip_unless_1n4d()
    def test_logical_slice_negative_start_stop_4dim_4d(test_case):
        var_shape = (30, 40, 20, 15)
        slice_tuples = [(10, None, 3), (1, -10, 4), (-15, -5, 2), (5, 11, 1)]
        arg_dict = OrderedDict()
        arg_dict["split_axis"] = list(range(4))
        arg_dict["device_tag"] = ["cpu", "gpu"]
        arg_dict["flow_dtype"] = [flow.float]
        arg_dict["device_num"] = [4]
        for arg in GenArgDict(arg_dict):
            _test_logical_slice(test_case, var_shape, slice_tuples, **arg)

    @flow.unittest.skip_unless_1n4d()
    def test_logical_slice_2dim_3d(test_case):
        var_shape = (30, 40)
        slice_tuples = [(10, 20, 3), (1, 30, 4)]
        arg_dict = OrderedDict()
        arg_dict["split_axis"] = list(range(2))
        arg_dict["device_tag"] = ["cpu", "gpu"]
        arg_dict["flow_dtype"] = [flow.float]
        arg_dict["device_num"] = [3]
        for arg in GenArgDict(arg_dict):
            _test_logical_slice(test_case, var_shape, slice_tuples, **arg)
