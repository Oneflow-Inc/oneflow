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


def run_leaky_relu(device_type, x_shape, data_type, alpha, use_deco=True):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    exe_config = flow.ExecutionConfig()
    exe_config.default_data_type(flow.float)

    leaky_relu_graph_func = None

    def func():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=type_name_to_flow_type[data_type],
                initializer=flow.constant_initializer(-0.5),
                trainable=True,
            )
            loss = flow.nn.leaky_relu(x, alpha=alpha)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    if use_deco:

        @flow.compiler.trace(type="train", execution_config=exe_config)
        def LeakyReluFunc():
            return func()

        leaky_relu_graph_func = LeakyReluFunc
    else:
        leaky_relu_graph_func = flow.compiler.trace(
            func, type="train", execution_config=exe_config
        )

    of_out = leaky_relu_graph_func().get()
    loss_diff = test_global_storage.Get("loss_diff")
    assert np.allclose(of_out.numpy(), np.full(x_shape, -0.05), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        test_global_storage.Get("x_diff"), np.full(x_shape, 0.1), rtol=1e-5, atol=1e-5
    )


@flow.unittest.skip_unless_1n1d()
class TestTrace(flow.unittest.TestCase):
    def test_trace_decorator(test_case):
        run_leaky_relu("cpu", (1, 2), "float32", 0.1, True)

    def test_trace_function(test_case):
        run_leaky_relu("cpu", (1, 2), "float32", 0.1, False)


if __name__ == "__main__":
    unittest.main()
