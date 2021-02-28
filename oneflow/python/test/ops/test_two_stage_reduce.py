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
import oneflow.typing as oft
import os

from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type

import test_global_storage


def _compare_with_numpy(test_case, np_func, x, y, axis, keepdims=True):
    x = test_global_storage.Get("x")
    dx = test_global_storage.Get("x_diff")
    np_y = np_func(x, axis=axis, keepdims=True)
    test_case.assertTrue(np.allclose(y, np_y, rtol=1e-5, atol=1e-5))
    mask = np.where(x == y, 1, 0)
    count = np.add.reduce(mask, axis=axis, keepdims=True)
    np_dx = np.where(x == y, 1 / count, 0)
    test_case.assertTrue(np.allclose(dx, np_dx, rtol=1e-5, atol=1e-5))


def _test_two_stage_reduce(
    test_case, flow_func, np_func, device_type, axis, split_axis
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(type="train", function_config=func_config)
    def two_stage_reduce_job(x: oft.Numpy.Placeholder((4, 20, 20, 20))):
        with flow.scope.placement(device_type, "0:0"):
            x += flow.get_variable(
                name="v1",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
        with flow.scope.placement(device_type, "0:0-3"):
            loss = flow_func(
                x.with_distribute(flow.distribute.split(split_axis)),
                axis=axis,
                keepdims=True,
            )
            loss = flow.identity(loss)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            return loss

    x = np.random.randint(low=0, high=10, size=(4, 20, 20, 20)).astype(np.float32)
    y = two_stage_reduce_job(x).get().numpy()
    _compare_with_numpy(test_case, np_func, x, y, axis=tuple(axis))


@flow.unittest.skip_unless_1n4d()
class TestTwoStageReduce(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_two_stage_reduce_max(test_case):
        arg_dict = OrderedDict()
        arg_dict["flow_func"] = [flow.math.two_stage_reduce_max]
        arg_dict["np_func"] = [np.maximum.reduce]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["axis"] = [[1], [1, 2], [1, 2, 3]]
        arg_dict["split_axis"] = [1]

        for arg in GenArgList(arg_dict):
            _test_two_stage_reduce(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_two_stage_reduce_min(test_case):
        arg_dict = OrderedDict()
        arg_dict["flow_func"] = [flow.math.two_stage_reduce_min]
        arg_dict["np_func"] = [np.minimum.reduce]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["axis"] = [[1], [1, 2], [1, 2, 3]]
        arg_dict["split_axis"] = [1]

        for arg in GenArgList(arg_dict):
            _test_two_stage_reduce(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
