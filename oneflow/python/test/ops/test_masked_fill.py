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
import os

from test_util import (
    GenArgDict,
    test_global_storage,
    type_name_to_flow_type,
    type_name_to_np_type,
)
import oneflow.typing as oft


def _masked_fill_np_fw_bw(x, mask, y_diff, type_name, value=0):
    brocadcast_shape = np.broadcast(x, mask).shape
    brocadcasted_x = np.broadcast_to(x, brocadcast_shape).astype(type_name)
    brocadcasted_mask = np.broadcast_to(mask, brocadcast_shape)
    masked_x = np.ma.array(brocadcasted_x, mask=brocadcasted_mask, fill_value=value)
    y = masked_x.filled()

    zero_like = np.zeros_like(y_diff)
    filted_y_diff = np.where(brocadcasted_mask, zero_like, y_diff)
    extended_axes_num = len(y_diff.shape) - len(x.shape)
    extended_axes = tuple(range(extended_axes_num))
    mid_diff = np.add.reduce(filted_y_diff, axis=extended_axes)
    diff_axes = list()
    for i in range(len(x.shape)):
        if x.shape[i] != y_diff.shape[i + extended_axes_num]:
            assert x.shape[i] == 1 and y_diff.shape[i + extended_axes_num] != 1
            diff_axes.append(i)
    if len(diff_axes) != 0:
        x_diff = np.add.reduce(mid_diff, axis=tuple(diff_axes), keepdims=True)
    else:
        x_diff = mid_diff

    return y, x_diff


def _test_masked_fill_fw_bw(test_case, device, x_shape, mask_shape, type_name, value=0):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    if type_name == "float16":
        flow_type = flow.float
        np_type = np.float32
    else:
        flow_type = type_name_to_flow_type[type_name]
        np_type = type_name_to_np_type[type_name]

    func_config.default_data_type(flow_type)

    @flow.global_function(type="train", function_config=func_config)
    def test_masked_fill_fw_bw_job(
        x: oft.Numpy.Placeholder(x_shape, dtype=flow_type),
        mask: oft.Numpy.Placeholder(mask_shape, dtype=flow_type),
    ):
        with flow.scope.placement(device, "0:0"):
            y = flow.get_variable(
                name="vx",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            x += flow.cast(y, flow_type)
            mask = flow.cast(mask, dtype=flow.int8)
            if type_name == "float16":
                out = flow.cast(
                    flow.masked_fill(flow.cast(x, flow.float16), mask, value),
                    flow.float,
                )
            else:
                out = flow.masked_fill(x, mask, value)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(out)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(out, test_global_storage.Setter("out"))
            flow.watch_diff(out, test_global_storage.Setter("out_diff"))
            return out

    x = np.random.randint(low=0, high=100, size=x_shape)
    mask = np.random.randint(low=0, high=2, size=mask_shape)

    test_masked_fill_fw_bw_job(x.astype(np_type), mask.astype(np_type)).get()
    out_diff = test_global_storage.Get("out_diff")

    np_out, np_x_diff = _masked_fill_np_fw_bw(x, mask, out_diff, np_type, value)

    if type_name == "float16":
        tolerance = 1e-3
    else:
        tolerance = 1e-5

    test_case.assertTrue(
        np.allclose(
            np_out, test_global_storage.Get("out"), rtol=tolerance, atol=tolerance
        )
    )
    test_case.assertTrue(
        np.allclose(
            np_x_diff, test_global_storage.Get("x_diff"), rtol=tolerance, atol=tolerance
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestMaskedFill(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_masked_fill_fw_bw(test_case):
        arg_dict = OrderedDict()
        arg_dict["type_name"] = [
            "float32",
            "float16",
            "double",
            "int8",
            "int32",
            "int64",
        ]
        arg_dict["device"] = ["gpu", "cpu"]
        arg_dict["x_shape"] = [
            (2, 2, 4),
            (2, 1, 4),
            (2, 2, 3, 2, 4),
        ]
        arg_dict["mask_shape"] = [(2, 1, 2, 4)]
        arg_dict["value"] = [2.5, -5.5]
        for arg in GenArgDict(arg_dict):
            if arg["device"] == "cpu" and arg["type_name"] == "float16":
                continue
            _test_masked_fill_fw_bw(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
