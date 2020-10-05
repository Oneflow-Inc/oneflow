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


def make_job(input_shape, norm_axis, params_axis, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def layer_norm_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.layers.layer_norm(
            x, begin_norm_axis=norm_axis, begin_params_axis=params_axis
        )

    return layer_norm_job


def make_xla_job(input_shape, norm_axis, params_axis, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def xla_layer_norm_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.layers.layer_norm(
            x, begin_norm_axis=norm_axis, begin_params_axis=params_axis
        )

    return xla_layer_norm_job


class TestLayerNorm(unittest.TestCase):
    def _test_body(self, x, norm_axis, params_axis, dtype=np.float32):
        f1 = make_job(x.shape, norm_axis, params_axis, dtype=flow.float32)
        f2 = make_xla_job(x.shape, norm_axis, params_axis, dtype=flow.float32)
        check_point = flow.train.CheckPoint()
        check_point.init()
        a = f1(x).get()
        b = f2(x).get()
        print("without xla: ", a.numpy())
        print("with xla", b.numpy())
        self.assertTrue(
            np.allclose(a.numpy(), b.numpy(), rtol=5e-3, atol=5e-3),
            a.numpy() - b.numpy(),
        )
        flow.clear_default_session()

    def _test_ones_body(self, shape, norm_axis=-1, params_axis=-1, dtype=np.float32):
        x = np.ones(shape, dtype=dtype)
        self._test_body(x, norm_axis, params_axis, dtype=dtype)

    def _test_random_body(self, shape, norm_axis=-1, params_axis=-1, dtype=np.float32):
        x = (10 * np.random.random(shape)).astype(dtype)
        self._test_body(x, norm_axis, params_axis, dtype=dtype)

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
