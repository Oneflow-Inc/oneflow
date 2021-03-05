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
import oneflow.typing as tp

flow.config.enable_debug_mode(True)
flow.config.enable_legacy_model_io(True)


def flow_relu(x):
    def make_job(input_shape, dtype=flow.float32):
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("fakedevice", "0:0"))

        @flow.global_function(type="predict", function_config=config)
        def relu_job(x: tp.Numpy.Placeholder(input_shape)) -> tp.Numpy:
            return flow.math.relu(x)

        return relu_job

    relu_fakedev_job = make_job(x.shape, dtype=flow.float32)
    y = relu_fakedev_job(x)
    return y


def np_relu(x):
    return np.where(x > 0, x, 0)


def _compare_with_np(test_case, input_shape):
    x = np.random.random(input_shape).astype(np.float32)
    np_res = np_relu(x)
    flow_res = flow_relu(x)
    test_case.assertTrue(np.array_equal(np_res, flow_res))


@flow.unittest.skip_unless_1n1d()
class TestRelu(flow.unittest.TestCase):
    def test_random_value(test_case):
        _compare_with_np(test_case, (2, 3))


if __name__ == "__main__":
    unittest.main()
