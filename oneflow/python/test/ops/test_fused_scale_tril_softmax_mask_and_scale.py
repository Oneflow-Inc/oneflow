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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type


def compare_with_not_fused(
    test_case, device_type, x_shape, data_type, diagonal, fill_value, scale, rate, seed
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    if data_type == "float16":
        dtype = flow.float
    else:
        dtype = type_name_to_flow_type[data_type]

    @flow.global_function(type="train", function_config=func_config)
    def test_fused_scale_tril_softmax_dropout_fw_bw_job():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=-1.0, maxval=1.0),
                trainable=True,
            )
            flow.watch(x, test_global_storage.Setter("x"))

            x1 = flow.identity(x)
            x2 = flow.identity(x)

            flow.watch_diff(x1, test_global_storage.Setter("x1_diff"))
            flow.watch_diff(x2, test_global_storage.Setter("x2_diff"))
            if data_type == "float16":
                y1 = flow.cast(
                    flow.nn.dropout(
                        flow.nn.softmax(
                            flow.math.fused_scale_tril(
                                flow.cast(x1, dtype=flow.float16),
                                diagonal=diagonal,
                                fill_value=fill_value,
                                scale=scale,
                            ),
                        ),
                        rate=rate,
                        seed=seed,
                        name="dropout",
                    ),
                    dtype=flow.float,
                )
                y2 = flow.cast(
                    flow.nn.fused_scale_tril_softmax_dropout(
                        flow.cast(x2, dtype=flow.float16),
                        diagonal=diagonal,
                        fill_value=fill_value,
                        scale=scale,
                        rate=rate,
                        seed=seed,
                    ),
                    dtype=flow.float,
                )
            else:
                y1 = flow.nn.dropout(
                    flow.nn.softmax(
                        flow.math.fused_scale_tril(
                            x1, diagonal=diagonal, fill_value=fill_value, scale=scale
                        )
                    ),
                    rate=rate,
                    seed=seed,
                    name="dropout",
                )
                y2 = flow.nn.fused_scale_tril_softmax_dropout(
                    x2,
                    diagonal=diagonal,
                    fill_value=fill_value,
                    scale=scale,
                    rate=rate,
                    seed=seed,
                )
            flow.watch(y1, test_global_storage.Setter("y1"))
            flow.watch(y2, test_global_storage.Setter("y2"))
            flow.watch_diff(y1, test_global_storage.Setter("y1_diff"))
            flow.watch_diff(y2, test_global_storage.Setter("y2_diff"))

            loss = y1 + y2
            total_loss = loss * x
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0.001]), momentum=0
        ).minimize(flow.math.reduce_sum(total_loss))

        return loss

    of_out = test_fused_scale_tril_softmax_dropout_fw_bw_job().get()

    y1 = test_global_storage.Get("y1")
    y2 = test_global_storage.Get("y2")

    tol = 1e-3 if data_type == "float16" else 1e-5
    test_case.assertTrue(np.allclose(y1, y2, rtol=tol, atol=tol, equal_nan=True))
    x1_diff = test_global_storage.Get("x1_diff")
    x2_diff = test_global_storage.Get("x2_diff")
    test_case.assertTrue(
        np.allclose(x1_diff, x2_diff, rtol=tol, atol=tol, equal_nan=True)
    )


@flow.unittest.skip_unless_1n1d()
class TestFusedScaleTrilSoftmaxDropout(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_fused_scale_tril_softmax_dropout(test_case):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [
            (2, 2, 5, 5),
            (10, 20),
            (32, 12, 128),
            (10, 960),
        ]
        arg_dict["data_type"] = ["float16", "float32", "double"]
        arg_dict["diagonal"] = [-1, 0]
        arg_dict["fill_value"] = [float("-inf"), 0]
        arg_dict["scale"] = [0.125]
        arg_dict["rate"] = [0.5]
        arg_dict["seed"] = [12345]
        for arg in GenArgList(arg_dict):
            if arg[0] == "cpu" and arg[2] == "float16":
                continue
            compare_with_not_fused(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
