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
import numpy as np
import oneflow as flow
from test_util import (
    GenArgDict,
    test_global_storage,
    type_name_to_flow_type,
    type_name_to_np_type,
)
import oneflow.typing as oft


def _test_tril_fw_bw(test_case, device, shape, type_name, diagonal=0):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    np_type = type_name_to_np_type[type_name]
    flow_type = type_name_to_flow_type[type_name]

    @flow.global_function(type="train", function_config=func_config)
    def test_tril_fw_bw_job(x: oft.Numpy.Placeholder(shape, dtype=flow.float),):
        with flow.scope.placement(device, "0:0"):
            x += flow.get_variable(
                name="vx",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            x = flow.cast(x, dtype=flow_type)
            out = flow.math.tril(x, diagonal)
            out = flow.cast(out, dtype=flow.float)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(out)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(out, test_global_storage.Setter("out"))
            flow.watch_diff(out, test_global_storage.Setter("out_diff"))
            return out

    check_point = flow.train.CheckPoint()
    check_point.init()
    x = np.random.randint(low=0, high=100, size=shape).astype(np.float32)
    test_tril_fw_bw_job(x).get()
    test_case.assertTrue(
        np.allclose(
            np.tril(test_global_storage.Get("x"), diagonal),
            test_global_storage.Get("out"),
        )
    )
    test_case.assertTrue(
        np.allclose(
            np.tril(test_global_storage.Get("out_diff"), diagonal),
            test_global_storage.Get("x_diff"),
        )
    )


def test_tril_fw_bw(test_case):
    arg_dict = OrderedDict()
    arg_dict["device"] = ["gpu", "cpu"]
    arg_dict["shape"] = [(6, 6), (3, 6, 8), (3, 4, 8, 6), (5, 3, 4, 8, 6)]
    arg_dict["type_name"] = ["float32", "double", "int8", "int32", "int64"]
    arg_dict["diagonal"] = [0, 1, -1]
    for arg in GenArgDict(arg_dict):
        _test_tril_fw_bw(test_case, **arg)
