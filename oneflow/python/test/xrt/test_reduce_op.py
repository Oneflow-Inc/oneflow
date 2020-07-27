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


class TestReduce(unittest.TestCase):
    run_test = False

    def _test_body(self, x, axis, keepdims, dtype=np.float32):
        if not self.run_test:
            return
        f1 = self.make_job(x.shape, axis, keepdims, dtype=flow.float32)
        f2 = self.make_xla_job(x.shape, axis, keepdims, dtype=flow.float32)
        a = f1(x).get()
        b = f2(x).get()
        print("without xla: ", a)
        print("with xla: ", b)
        self.assertTrue(a.shape == b.shape)
        self.assertTrue(np.allclose(a.numpy(), b.numpy(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

        f3 = self.make_trt_job(x.shape, axis, keepdims, dtype=flow.float32)
        c = f3(x).get()
        print("with tensorrt: ", c)
        self.assertTrue(a.shape == c.shape)
        self.assertTrue(np.allclose(a.numpy(), c.numpy(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, axis, keepdims, dtype=np.float32):
        x = np.ones(shape, dtype=dtype)
        self._test_body(x, axis, keepdims, dtype=dtype)

    def _test_random_body(self, shape, axis, keepdims, dtype=np.float32):
        x = np.random.random(shape).astype(dtype)
        self._test_body(x, axis, keepdims, dtype=dtype)

    def test_ones_input(self):
        # self._test_ones_body((1), [0], False)
        self._test_ones_body((1), [0], True)
        self._test_ones_body((1, 10), [1], False)
        self._test_ones_body((1, 10), [1], True)
        # self._test_ones_body((1, 10), [0, 1], False)
        self._test_ones_body((1, 10), [0, 1], True)
        self._test_ones_body((2, 10, 2), [1, 2], False)
        self._test_ones_body((2, 10, 2), [1, 2], True)

    def test_random_input(self):
        # self._test_random_body((1), [0], False)
        self._test_random_body((1), [0], True)
        self._test_random_body((1, 10), [1], False)
        self._test_random_body((1, 10), [1], True)
        # self._test_random_body((1, 10), [0, 1], False)
        self._test_random_body((1, 10), [0, 1], True)
        self._test_random_body((2, 10, 2), [1, 2], False)
        self._test_random_body((2, 10, 2), [1, 2], True)


class TestReduceSum(TestReduce):
    run_test = True

    def make_job(self, x_shape, axis, keepdims, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def reduce_sum_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.math.reduce_sum(x, axis=axis, keepdims=keepdims)

        return reduce_sum_job

    def make_xla_job(self, x_shape, axis, keepdims, dtype=flow.float32):
        config.use_xla_jit(True)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def xla_reduce_sum_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.math.reduce_sum(x, axis=axis, keepdims=keepdims)

        return xla_reduce_sum_job

    def make_trt_job(self, x_shape, axis, keepdims, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(True)

        @flow.global_function(config)
        def trt_reduce_sum_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.math.reduce_sum(x, axis=axis, keepdims=keepdims)

        return trt_reduce_sum_job


# XLA has not support ReduceMean, so it will fallback to oneflow automatically.
class TestReduceMean(TestReduce):
    run_test = True

    def make_job(self, x_shape, axis, keepdims, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def reduce_mean_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.math.reduce_mean(x, axis=axis, keepdims=keepdims)

        return reduce_mean_job

    def make_xla_job(self, x_shape, axis, keepdims, dtype=flow.float32):
        config.use_xla_jit(True)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def xla_reduce_mean_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.math.reduce_mean(x, axis=axis, keepdims=keepdims)

        return xla_reduce_mean_job

    def make_trt_job(self, x_shape, axis, keepdims, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(True)

        @flow.global_function(config)
        def trt_reduce_mean_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.math.reduce_mean(x, axis=axis, keepdims=keepdims)

        return trt_reduce_mean_job


if __name__ == "__main__":
    unittest.main()
