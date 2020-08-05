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

config = flow.function_config()


def make_job(input_shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def sigmoid_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.math.sigmoid(x)

    return sigmoid_job


def make_xla_job(input_shape, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def xla_sigmoid_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.math.sigmoid(x)

    return xla_sigmoid_job


def make_trt_job(input_shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(True)

    @flow.global_function(config)
    def trt_sigmoid_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.math.sigmoid(x)

    return trt_sigmoid_job


class TestSigmoid(unittest.TestCase):
    def _test_body(self, x, dtype=np.float32):
        f1 = make_job(x.shape, dtype=flow.float32)
        f2 = make_xla_job(x.shape, dtype=flow.float32)
        a = f1(x).get()
        b = f2(x).get()
        print("without xla: ", a)
        print("with xla: ", b)
        self.assertTrue(np.allclose(a.numpy(), b.numpy(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

        f3 = make_trt_job(x.shape, dtype=flow.float32)
        c = f3(x).get()
        print("with tensorrt: ", c)
        self.assertTrue(np.allclose(a.numpy(), c.numpy(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, dtype=np.float32):
        x = np.ones(shape, dtype=dtype)
        self._test_body(x, dtype=dtype)

    def _test_random_body(self, shape, dtype=np.float32):
        x = np.random.random(shape).astype(dtype)
        self._test_body(x, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1))
        self._test_ones_body((1, 10))
        self._test_ones_body((2, 10, 2))
        self._test_ones_body((2, 5, 2, 2))

    def test_random_input(self):
        self._test_random_body((1))
        self._test_random_body((1, 10))
        self._test_random_body((2, 10, 2))
        self._test_random_body((2, 5, 2, 2))


if __name__ == "__main__":
    unittest.main()
