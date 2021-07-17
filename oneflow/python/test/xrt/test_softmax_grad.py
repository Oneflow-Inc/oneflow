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


def make_job(shape, axis, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def softmax_grad_job(
        y=flow.FixedTensorDef(shape, dtype=dtype),
        dy=flow.FixedTensorDef(shape, dtype=dtype),
    ):
        return flow.nn.softmax_grad(y, dy, axis=axis)

    return softmax_grad_job


def make_xla_job(shape, axis, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def xla_softmax_grad_job(
        y=flow.FixedTensorDef(shape, dtype=dtype),
        dy=flow.FixedTensorDef(shape, dtype=dtype),
    ):
        return flow.nn.softmax_grad(y, dy, axis=axis)

    return xla_softmax_grad_job


class TestSoftmaxGrad(unittest.TestCase):
    def _test_body(self, y, dy, axis, dtype=np.float32):
        f1 = make_job(y.shape, axis, dtype=flow.float32)
        f2 = make_xla_job(y.shape, axis, dtype=flow.float32)
        a = f1(y, dy).get()
        b = f2(y, dy).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(a.shape == b.shape)
        self.assertTrue(np.allclose(a.numpy(), b.numpy(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, axis, dtype=np.float32):
        y = np.ones(shape, dtype=dtype)
        dy = np.ones(shape, dtype=dtype)
        self._test_body(y, dy, axis, dtype=dtype)

    def _test_random_body(self, shape, axis, dtype=np.float32):
        y = np.random.random(shape).astype(dtype)
        dy = np.random.random(shape).astype(dtype)
        self._test_body(y, dy, axis, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((2, 5), axis=1)
        self._test_ones_body((2, 5), axis=-1)
        self._test_ones_body((1, 5, 2), axis=1)
        self._test_ones_body((1, 5, 2), axis=2)

    def test_random_input(self):
        self._test_random_body((2, 5), axis=1)
        self._test_random_body((2, 5), axis=-1)
        self._test_random_body((1, 5, 2), axis=1)
        self._test_random_body((1, 5, 2), axis=2)


if __name__ == "__main__":
    unittest.main()
