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


def make_job(input_shape, axis, fuse_add_to_output, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)
    config.enable_fuse_add_to_output(fuse_add_to_output)

    @flow.global_function(config)
    def batch_norm_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        out = flow.layers.batch_normalization(x, axis=axis)
        c = flow.get_variable(
            "c",
            shape=out.shape,
            dtype=flow.float,
            initializer=flow.ones_initializer(),
            trainable=True,
        )
        out = flow.math.add_n([out, c])
        return out

    return batch_norm_job


def make_xla_job(input_shape, axis, fuse_add_to_output, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)
    config.enable_fuse_add_to_output(fuse_add_to_output)

    @flow.global_function(config)
    def xla_batch_norm_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        out = flow.layers.batch_normalization(x, axis=axis)
        c = flow.get_variable(
            "c",
            shape=out.shape,
            dtype=flow.float,
            initializer=flow.ones_initializer(),
            trainable=True,
        )
        out = flow.math.add_n([out, c])
        return out

    return xla_batch_norm_job


def make_trt_job(input_shape, axis, fuse_add_to_output, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(True)
    config.enable_fuse_add_to_output(fuse_add_to_output)

    @flow.global_function(config)
    def trt_batch_norm_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        out = flow.layers.batch_normalization(x, axis=axis)
        c = flow.get_variable(
            "c",
            shape=out.shape,
            dtype=flow.float,
            initializer=flow.ones_initializer(),
            trainable=True,
        )
        out = flow.math.add_n([out, c])
        return out

    return trt_batch_norm_job


class TestRelu(unittest.TestCase):
    def _test_body(self, x, axis, fuse_add_to_output, dtype=np.float32):
        f1 = make_job(x.shape, axis, fuse_add_to_output, dtype=flow.float32)
        f2 = make_xla_job(x.shape, axis, fuse_add_to_output, dtype=flow.float32)
        f3 = make_trt_job(x.shape, axis, fuse_add_to_output, dtype=flow.float32)
        check_point = flow.train.CheckPoint()
        check_point.init()
        a = f1(x).get()
        b = f2(x).get()
        print("without xla: ", a.numpy())
        print("with xla: ", b.numpy())
        self.assertTrue(np.allclose(a.numpy(), b.numpy(), rtol=1e-03, atol=1e-05))
        c = f3(x).get()
        print("with tensorrt: ", c.numpy())
        self.assertTrue(np.allclose(a.numpy(), c.numpy(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, axis, fuse_add_to_output, dtype=np.float32):
        x = np.ones(shape, dtype=dtype)
        self._test_body(x, axis, fuse_add_to_output, dtype=dtype)

    def _test_random_body(self, shape, axis, fuse_add_to_output, dtype=np.float32):
        x = np.random.random(shape).astype(dtype)
        self._test_body(x, axis, fuse_add_to_output, dtype=dtype)

    """
      TensorRT batch norm only support 4-d tensor (NCHW).
    """

    def test_ones_input(self):
        self._test_ones_body((2, 1, 2, 2), 1, True)
        self._test_ones_body((2, 1, 2, 2), 1, False)
        self._test_ones_body((2, 5, 2, 2), 1, True)
        self._test_ones_body((2, 5, 2, 2), 1, False)

    def test_random_input(self):
        self._test_random_body((2, 1, 2, 2), 1, True)
        self._test_random_body((2, 1, 2, 2), 1, False)
        self._test_random_body((2, 5, 2, 2), 1, True)
        self._test_random_body((2, 5, 2, 2), 1, False)


if __name__ == "__main__":
    unittest.main()
