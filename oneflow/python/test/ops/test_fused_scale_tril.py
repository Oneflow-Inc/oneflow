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
from test_util import (
    GenArgDict,
    test_global_storage,
    type_name_to_flow_type,
    type_name_to_np_type,
)
import oneflow.typing as oft
import os


def _test_fused_scale_tril_fw_bw(
    test_case, device, shape, type_name, diagonal, fill_value, scale
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    if type_name == "float16":
        flow_type = flow.float
        np_type = np.float32
    else:
        flow_type = type_name_to_flow_type[type_name]
        np_type = type_name_to_np_type[type_name]

    @flow.global_function(type="train", function_config=func_config)
    def test_fused_scale_tril_fw_bw_job(
        x: oft.Numpy.Placeholder(shape, dtype=flow_type),
    ):
        with flow.scope.placement(device, "0:0"):
            x_var = flow.get_variable(
                name="xv",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            x += flow.cast(x_var, dtype=flow_type)
            if type_name == "float16":
                out = flow.cast(
                    flow.math.fused_scale_tril(
                        flow.cast(x, flow.float16), diagonal, scale=scale
                    ),
                    flow.float,
                )
            else:
                out = flow.math.fused_scale_tril(x, diagonal, scale=scale)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(out)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(out, test_global_storage.Setter("out"))
            flow.watch_diff(out, test_global_storage.Setter("out_diff"))
            return out

    x = np.random.randint(low=0, high=100, size=shape)
    test_fused_scale_tril_fw_bw_job(x.astype(np_type)).get()

    np_out = np.where(
        np.tril(np.ones(shape), diagonal),
        test_global_storage.Get("x") * scale,
        np.full(shape, fill_value).astype(np_type),
    )
    np_x_diff = np.tril(test_global_storage.Get("out_diff"), diagonal) * scale

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
class TestFusedScaleTril(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_fused_scale_tril_fw_bw(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["gpu"]
        arg_dict["type_name"] = [
            "float32",
            "float16",
            "double",
            "int32",
            "int64",
        ]
        arg_dict["shape"] = [(3, 6, 8)]
        arg_dict["diagonal"] = [-8, -1, 0, 8]
        arg_dict["fill_value"] = [1.0, 0]
        arg_dict["scale"] = [5.0, 3]
        for arg in GenArgDict(arg_dict):
            if isinstance(arg["fill_value"], float) and arg_dict["type_name"] not in [
                "float32",
                "float16",
                "double",
            ]:
                continue
            _test_fused_scale_tril_fw_bw(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
