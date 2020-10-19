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


def make_job(shape, mean_shape, norm_axis, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def layer_norm_grad_job(
        dy=flow.FixedTensorDef(shape, dtype=dtype),
        x=flow.FixedTensorDef(shape, dtype=dtype),
        mean=flow.FixedTensorDef(mean_shape, dtype=dtype),
        inv_variance=flow.FixedTensorDef(mean_shape, dtype=dtype),
    ):
        return flow.layers.layer_norm_grad(
            dy, x, mean, inv_variance, begin_norm_axis=norm_axis
        )

    return layer_norm_grad_job


def make_xla_job(shape, mean_shape, norm_axis, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def xla_layer_norm_grad_job(
        dy=flow.FixedTensorDef(shape, dtype=dtype),
        x=flow.FixedTensorDef(shape, dtype=dtype),
        mean=flow.FixedTensorDef(mean_shape, dtype=dtype),
        inv_variance=flow.FixedTensorDef(mean_shape, dtype=dtype),
    ):
        return flow.layers.layer_norm_grad(
            dy, x, mean, inv_variance, begin_norm_axis=norm_axis
        )

    return xla_layer_norm_grad_job


class TestLayerNormGrad(unittest.TestCase):
    def _test_body(self, dy, x, mean, inv_variance, norm_axis, dtype=np.float32):
        f1 = make_job(x.shape, mean.shape, norm_axis, dtype=flow.float32)
        f2 = make_xla_job(x.shape, mean.shape, norm_axis, dtype=flow.float32)
        a = f1(dy, x, mean, inv_variance).get()
        b = f2(dy, x, mean, inv_variance).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(np.allclose(a.numpy(), b.numpy(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, norm_axis=-1, dtype=np.float32):
        dy = np.ones(shape, dtype=dtype)
        x = np.ones(shape, dtype=dtype)
        if norm_axis < 0:
            norm_axis += len(shape)
        mean_shape = shape[:norm_axis]
        mean = np.ones(mean_shape, dtype=dtype)
        inv_variance = np.ones(mean_shape, dtype=dtype)
        self._test_body(dy, x, mean, inv_variance, norm_axis, dtype=dtype)

    def _test_random_body(self, shape, norm_axis=-1, dtype=np.float32):
        dy = np.random.random(shape).astype(dtype)
        x = np.random.random(shape).astype(dtype)
        if norm_axis < 0:
            norm_axis += len(shape)
        mean_shape = shape[:norm_axis]
        mean = np.random.random(mean_shape).astype(dtype)
        inv_variance = np.random.random(mean_shape).astype(dtype)
        self._test_body(dy, x, mean, inv_variance, norm_axis, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1, 10))
        self._test_ones_body((2, 10, 2))
        self._test_ones_body((2, 5, 2, 2))

    def test_random_input(self):
        self._test_random_body((1, 10))
        self._test_random_body((2, 10, 2))
        self._test_random_body((2, 5, 2, 2))


if __name__ == "__main__":
    unittest.main()
