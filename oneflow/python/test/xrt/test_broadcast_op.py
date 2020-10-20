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


class TestBroadcastOp(unittest.TestCase):
    run_test = False

    def _test_body(self, x, y, dtype=np.float32):
        if not self.run_test:
            return
        f1 = self.make_job(x.shape, y.shape, dtype=flow.float32)
        f2 = self.make_xla_job(x.shape, y.shape, dtype=flow.float32)
        a = f1(x, y).get()
        b = f2(x, y).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(np.allclose(a.numpy(), b.numpy(), rtol=1e-03, atol=1e-05))

        flow.clear_default_session()

    def _test_ones_body(self, x_shape, y_shape, dtype=np.float32):
        x = np.ones(x_shape, dtype=dtype)
        y = np.ones(y_shape, dtype=dtype)
        self._test_body(x, y, dtype=dtype)

    def _test_random_body(self, x_shape, y_shape, dtype=np.float32):
        x = np.random.random(x_shape).astype(dtype)
        y = np.random.random(y_shape).astype(dtype)
        self._test_body(x, y, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1, 10), (1, 1))
        self._test_ones_body((2, 10, 2), (2, 1, 2))
        self._test_ones_body((2, 5, 2, 2), (1, 5, 2, 2))

    def test_random_input(self):
        self._test_random_body((1, 10), (1, 1))
        self._test_random_body((2, 10, 2), (2, 1, 2))
        self._test_random_body((2, 5, 2, 2), (1, 5, 2, 2))


class TestBroadcastAddOp(TestBroadcastOp):
    run_test = True

    def make_job(self, x_shape, y_shape, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def broadcast_add_job(
            x=flow.FixedTensorDef(x_shape, dtype=dtype),
            y=flow.FixedTensorDef(y_shape, dtype=dtype),
        ):
            return flow.math.add(x, y)

        return broadcast_add_job

    def make_xla_job(self, x_shape, y_shape, dtype=flow.float32):
        config.use_xla_jit(True)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def xla_broadcast_add_job(
            x=flow.FixedTensorDef(x_shape, dtype=dtype),
            y=flow.FixedTensorDef(y_shape, dtype=dtype),
        ):
            return flow.math.add(x, y)

        return xla_broadcast_add_job


class TestBroadcastMulOp(TestBroadcastOp):
    run_test = True

    def make_job(self, x_shape, y_shape, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def broadcast_mul_job(
            x=flow.FixedTensorDef(x_shape, dtype=dtype),
            y=flow.FixedTensorDef(y_shape, dtype=dtype),
        ):
            return flow.math.multiply(x, y)

        return broadcast_mul_job

    def make_xla_job(self, x_shape, y_shape, dtype=flow.float32):
        config.use_xla_jit(True)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def xla_broadcast_mul_job(
            x=flow.FixedTensorDef(x_shape, dtype=dtype),
            y=flow.FixedTensorDef(y_shape, dtype=dtype),
        ):
            return flow.math.multiply(x, y)

        return xla_broadcast_mul_job


class TestBroadcastDivOp(TestBroadcastOp):
    run_test = True

    def make_job(self, x_shape, y_shape, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def broadcast_div_job(
            x=flow.FixedTensorDef(x_shape, dtype=dtype),
            y=flow.FixedTensorDef(y_shape, dtype=dtype),
        ):
            return flow.math.divide(x, y)

        return broadcast_div_job

    def make_xla_job(self, x_shape, y_shape, dtype=flow.float32):
        config.use_xla_jit(True)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def xla_broadcast_div_job(
            x=flow.FixedTensorDef(x_shape, dtype=dtype),
            y=flow.FixedTensorDef(y_shape, dtype=dtype),
        ):
            return flow.math.divide(x, y)

        return xla_broadcast_div_job


class TestBroadcastMinOp(TestBroadcastOp):
    run_test = True

    def make_job(self, x_shape, y_shape, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def broadcast_min_job(
            x=flow.FixedTensorDef(x_shape, dtype=dtype),
            y=flow.FixedTensorDef(y_shape, dtype=dtype),
        ):
            return flow.math.minimum(x, y)

        return broadcast_min_job

    def make_xla_job(self, x_shape, y_shape, dtype=flow.float32):
        config.use_xla_jit(True)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def xla_broadcast_min_job(
            x=flow.FixedTensorDef(x_shape, dtype=dtype),
            y=flow.FixedTensorDef(y_shape, dtype=dtype),
        ):
            return flow.math.minimum(x, y)

        return xla_broadcast_min_job


if __name__ == "__main__":
    unittest.main()
