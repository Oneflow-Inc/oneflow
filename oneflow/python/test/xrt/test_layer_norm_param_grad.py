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


def make_job(shape, gamma_shape, params_axis, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def layer_norm_param_grad_job(
        dy=flow.FixedTensorDef(shape, dtype=dtype),
        norm=flow.FixedTensorDef(shape, dtype=dtype),
        gamma=flow.FixedTensorDef(gamma_shape, dtype=dtype),
    ):
        return flow.layers.layer_norm_param_grad(
            dy, norm, gamma, begin_params_axis=params_axis
        )

    return layer_norm_param_grad_job


def make_xla_job(shape, gamma_shape, params_axis, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def xla_layer_norm_param_grad_job(
        dy=flow.FixedTensorDef(shape, dtype=dtype),
        norm=flow.FixedTensorDef(shape, dtype=dtype),
        gamma=flow.FixedTensorDef(gamma_shape, dtype=dtype),
    ):
        return flow.layers.layer_norm_param_grad(
            dy, norm, gamma, begin_params_axis=params_axis
        )

    return xla_layer_norm_param_grad_job


class TestLayerNormParamGrad(unittest.TestCase):
    def _test_body(self, dy, norm, gamma, params_axis, dtype=np.float32):
        f1 = make_job(dy.shape, gamma.shape, params_axis, dtype=flow.float32)
        f2 = make_xla_job(dy.shape, gamma.shape, params_axis, dtype=flow.float32)
        (d_norm1, d_beta1, d_gamma1) = f1(dy, norm, gamma).get()
        (d_norm2, d_beta2, d_gamma2) = f2(dy, norm, gamma).get()

        print("normalize diff:")
        print("    without xla: ", d_norm1)
        print("    with xla: ", d_norm2)
        print("beta diff:")
        print("    without xla: ", d_beta1)
        print("    with xla: ", d_beta2)
        print("gamma diff:")
        print("    without xla: ", d_gamma1)
        print("    with xla: ", d_gamma2)

        self.assertTrue(d_norm1.shape, d_norm2.shape)
        self.assertTrue(d_beta1.shape, d_beta2.shape)
        self.assertTrue(d_gamma1.shape, d_gamma2.shape)

        self.assertTrue(
            np.allclose(d_norm1.numpy(), d_norm2.numpy(), rtol=1e-03, atol=1e-05)
        )
        self.assertTrue(
            np.allclose(d_beta1.numpy(), d_beta2.numpy(), rtol=1e-03, atol=1e-05)
        )
        self.assertTrue(
            np.allclose(d_gamma1.numpy(), d_gamma2.numpy(), rtol=1e-03, atol=1e-05)
        )

        flow.clear_default_session()

    def _test_ones_body(self, shape, params_axis=-1, dtype=np.float32):
        dy = np.ones(shape, dtype=dtype)
        norm = np.ones(shape, dtype=dtype)
        if params_axis < 0:
            params_axis += len(shape)
        gamma_shape = shape[params_axis:]
        if len(gamma_shape) == 0:
            gamma_shape = [1]
        gamma = np.ones(gamma_shape, dtype=dtype)
        self._test_body(dy, norm, gamma, params_axis, dtype=dtype)

    def _test_random_body(self, shape, params_axis=-1, dtype=np.float32):
        dy = np.random.random(shape).astype(dtype)
        norm = np.random.random(shape).astype(dtype)
        if params_axis < 0:
            params_axis += len(shape)
        gamma_shape = shape[params_axis:]
        if len(gamma_shape) == 0:
            gamma_shape = [1]
        gamma = np.random.random(gamma_shape).astype(dtype)
        self._test_body(dy, norm, gamma, params_axis, dtype=dtype)

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
