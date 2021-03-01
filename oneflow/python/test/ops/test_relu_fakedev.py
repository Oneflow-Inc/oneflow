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

config = flow.function_config()


def make_job(input_shape, dtype=flow.float32):
    @flow.global_function(type="predict", function_config=config)
    def relu_job(x: tp.Numpy.Placeholder(input_shape)) -> tp.Numpy:
        with flow.scope.placement("fakedevice", "0:0"):
            return flow.math.relu(x)

    return relu_job


@flow.unittest.skip_unless_1n1d()
class TestRelu(flow.unittest.TestCase):
    def test_body(test_case):
        x = np.random.random((2, 3)).astype(np.float32)
        relu_fakedev_job = make_job(x.shape, dtype=flow.float32)
        y = relu_fakedev_job(x)
        print(y)


if __name__ == "__main__":
    unittest.main()
