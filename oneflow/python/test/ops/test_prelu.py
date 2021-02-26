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
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft


def _check(test_case, x, y, shared_axes):
    alpha_of = test_global_storage.Get("alpha")
    alpha = np.expand_dims(alpha_of, axis=0)
    dy = test_global_storage.Get("loss_diff")
    np_prelu_out = np.where(x > 0, x, x * alpha)
    np_prelu_x_diff = np.where(x > 0, dy, dy * alpha)
    np_prelu_alpha_diff = np.where(x > 0, 0, dy * x)
    np_prelu_alpha_diff = np.add.reduce(
        np_prelu_alpha_diff, axis=shared_axes, keepdims=True
    )
    np_prelu_alpha_diff = np.add.reduce(np_prelu_alpha_diff, axis=0)
    test_case.assertTrue(np.allclose(np_prelu_out, y))
    test_case.assertTrue(
        np.allclose(np_prelu_x_diff, test_global_storage.Get("x_diff"))
    )
    test_case.assertTrue(
        np.allclose(np_prelu_alpha_diff, test_global_storage.Get("alpha_diff"))
    )


def _run_test(test_case, device_type, dtype, x_shape, shared_axes):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def PreluJob(
        x: oft.Numpy.Placeholder(x_shape, dtype=type_name_to_flow_type[dtype])
    ):
        with flow.scope.placement(device_type, "0:0"):
            x += flow.get_variable(
                name="v1",
                shape=(1,),
                dtype=type_name_to_flow_type[dtype],
                initializer=flow.zeros_initializer(),
            )
            loss = flow.layers.prelu(
                x,
                alpha_initializer=flow.random_uniform_initializer(
                    minval=0.1, maxval=0.9
                ),
                shared_axes=shared_axes,
                name="prelu",
            )
            alpha_shape = list(x.shape[1:])
            if shared_axes is not None:
                for i in shared_axes:
                    alpha_shape[i - 1] = 1
            alpha = flow.get_variable(
                "prelu-alpha",
                shape=tuple(alpha_shape),
                dtype=type_name_to_flow_type[dtype],
                initializer=flow.random_uniform_initializer(minval=0.1, maxval=0.9),
            )
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(alpha, test_global_storage.Setter("alpha"))
            flow.watch_diff(alpha, test_global_storage.Setter("alpha_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    x = (np.random.random(x_shape) - 1).astype(type_name_to_np_type[dtype])
    y = PreluJob(x).get()
    _check(test_case, x, y.numpy(), shared_axes)


@flow.unittest.skip_unless_1n1d()
class TestPrelu(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_prelu(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["dtype"] = ["float32"]
        arg_dict["x_shape"] = [(10, 32, 20, 20)]
        arg_dict["shared_axes"] = [(2,), (1, 2), (1, 3), (1, 2, 3)]

        for arg in GenArgList(arg_dict):
            _run_test(*arg)


if __name__ == "__main__":
    unittest.main()
