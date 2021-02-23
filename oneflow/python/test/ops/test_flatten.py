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
from test_util import GenArgList
import test_global_storage


def compare_with_numpy(test_case, device_type, input_shape, start_end_dim):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    start_dim = start_end_dim[0]
    end_dim = start_end_dim[1]

    @flow.global_function(type="train", function_config=func_config)
    def FlattenJob() -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "in",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=2, maxval=5),
                trainable=True,
            )

            loss = flow.flatten(x, start_dim=start_dim, end_dim=end_dim)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))

            return loss

    # OneFlow
    of_out = FlattenJob()

    # Numpy
    of_x = test_global_storage.Get("x")
    of_x_shape = of_x.shape
    of_x_diff = test_global_storage.Get("x_diff")

    true_end_dim = end_dim + len(of_x_shape) if end_dim < 0 else end_dim
    new_shape = []
    for i in range(0, start_dim):
        new_shape.append(of_x_shape[i])
    flatten_dim = 1
    for i in range(start_dim, true_end_dim + 1):
        flatten_dim *= of_x_shape[i]
    new_shape.append(flatten_dim)
    for i in range(true_end_dim + 1, len(of_x_shape)):
        new_shape.append(of_x_shape[i])

    np_out = np.reshape(of_x, tuple(new_shape))

    test_case.assertTrue(of_out.shape == np_out.shape)
    test_case.assertTrue(np.allclose(of_out, np_out, rtol=1e-5, atol=1e-5))
    test_case.assertTrue(
        np.allclose(of_x_diff, np.ones(of_x_diff.shape), rtol=1e-5, atol=1e-5)
    )


@flow.unittest.skip_unless_1n1d()
class TestFlatten(flow.unittest.TestCase):
    def test_flatten(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(2, 3, 4, 5)]
        arg_dict["start_end_dim"] = [(0, -1), (1, 3), (2, -2)]
        for arg in GenArgList(arg_dict):
            compare_with_numpy(*arg)


if __name__ == "__main__":
    unittest.main()
