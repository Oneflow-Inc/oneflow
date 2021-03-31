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
import tensorflow as tf
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(test_case, device_type, x_shape, data_type, axis):
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
                print("fp16")
                y1 = flow.cast(
                    flow.nn.softmax(
                        flow.math.fused_scale_tril(
                            flow.cast(x1, dtype=flow.float16), diagonal=0, scale=1.0
                        )
                    ),
                    dtype=flow.float,
                )
                y2 = flow.cast(
                    flow.nn.fused_scale_tril_softmax_dropout(
                        flow.cast(x2, dtype=flow.float16), diagonal=0, scale=1.0
                    ),
                    dtype=flow.float,
                )
            else:
                y1 = flow.nn.softmax(
                    flow.math.fused_scale_tril(x1, diagonal=0, scale=1.0)
                )
                y2 = flow.nn.fused_scale_tril_softmax_dropout(x2, diagonal=0, scale=1.0)
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

    # OneFlow
    print("start")

    of_out = test_fused_scale_tril_softmax_dropout_fw_bw_job().get()
    print("start")

    y1 = test_global_storage.Get("y1")
    y2 = test_global_storage.Get("y2")
    print("y1", y1.flatten()[0:20])
    print("y2", y2.flatten()[0:20])
    tol = 1e-3 if data_type == flow.float16 else 1e-5
    x1_diff = test_global_storage.Get("x1_diff")
    x2_diff = test_global_storage.Get("x2_diff")
    print("x1_diff", x1_diff.flatten()[0:20])
    print("x2_diff", x2_diff.flatten()[0:20])
    test_case.assertTrue(np.allclose(x1_diff, x2_diff, rtol=tol, atol=tol))
    print("end")


@flow.unittest.skip_unless_1n1d()
class TestSoftmax(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_softmax_shape(test_case):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [
            (10, 10, 20, 30),
            (10, 20, 13),
            (10, 20, 30),
            (10, 20),
            (10, 60),
            (32, 12, 128),
            (10, 960),
            (12, 2001),
            (10, 4096),
            (10, 8092),
            (256, 1001),
            (100, 65536),
            (10, 65535),
        ]
        arg_dict["data_type"] = ["float16", "float32"]
        arg_dict["axis"] = [-1]
        for arg in GenArgList(arg_dict):
            if arg[0] == "cpu" and arg[2] == "float16":
                continue
            print(*arg)
            compare_with_tensorflow(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
