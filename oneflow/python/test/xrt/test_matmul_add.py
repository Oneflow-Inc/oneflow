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


def make_job(
    a_shape,
    b_shape,
    trans_a=False,
    trans_b=False,
    fuse_add_to_output=True,
    dtype=flow.float32,
):
    config.use_xla_jit(False)
    config.use_tensorrt(False)
    config.enable_fuse_add_to_output(fuse_add_to_output)

    @flow.global_function(config)
    def matmul_job(
        a=flow.FixedTensorDef(a_shape, dtype=dtype),
        b=flow.FixedTensorDef(b_shape, dtype=dtype),
    ):
        out = flow.matmul(a, b, transpose_a=trans_a, transpose_b=trans_b)
        c = flow.get_variable(
            "c",
            shape=out.shape,
            dtype=flow.float,
            initializer=flow.ones_initializer(),
            trainable=True,
        )
        out = flow.math.add_n([out, c])
        return out

    return matmul_job


def make_xla_job(
    a_shape,
    b_shape,
    trans_a=False,
    trans_b=False,
    fuse_add_to_output=True,
    dtype=flow.float32,
):
    config.use_xla_jit(True)
    config.use_tensorrt(False)
    config.enable_fuse_add_to_output(fuse_add_to_output)

    @flow.global_function(config)
    def xla_matmul_job(
        a=flow.FixedTensorDef(a_shape, dtype=dtype),
        b=flow.FixedTensorDef(b_shape, dtype=dtype),
    ):
        out = flow.matmul(a, b, transpose_a=trans_a, transpose_b=trans_b)
        c = flow.get_variable(
            "c",
            shape=out.shape,
            dtype=flow.float,
            initializer=flow.ones_initializer(),
            trainable=True,
        )
        out = flow.math.add_n([out, c])
        return out

    return xla_matmul_job


def make_trt_job(
    a_shape,
    b_shape,
    trans_a=False,
    trans_b=False,
    fuse_add_to_output=True,
    dtype=flow.float32,
):
    config.use_xla_jit(False)
    config.use_tensorrt(True)
    config.enable_fuse_add_to_output(fuse_add_to_output)

    @flow.global_function(config)
    def trt_matmul_job(
        a=flow.FixedTensorDef(a_shape, dtype=dtype),
        b=flow.FixedTensorDef(b_shape, dtype=dtype),
    ):
        out = flow.matmul(a, b, transpose_a=trans_a, transpose_b=trans_b)
        c = flow.get_variable(
            "c",
            shape=out.shape,
            dtype=flow.float,
            initializer=flow.ones_initializer(),
            trainable=True,
        )
        out = flow.math.add_n([out, c])
        return out

    return trt_matmul_job


class TestMatmul(unittest.TestCase):
    def make_shape(self, m, n, transpose):
        if transpose:
            return (n, m)
        else:
            return (m, n)

    def _test_body(self, a, b, trans_a, trans_b, fuse_add_to_output, dtype=np.float32):
        f1 = make_job(a.shape, b.shape, trans_a, trans_b, fuse_add_to_output)
        f2 = make_xla_job(a.shape, b.shape, trans_a, trans_b, fuse_add_to_output)
        f3 = make_trt_job(a.shape, b.shape, trans_a, trans_b, fuse_add_to_output)
        x = f1(a, b).get()
        y = f2(a, b).get()
        z = f3(a, b).get()
        print("without xla: ", x.numpy())
        print("with xla: ", y.numpy())
        print("with tensorrt: ", z.numpy())
        self.assertTrue(np.allclose(x.numpy(), y.numpy(), rtol=1e-03, atol=1e-05))
        self.assertTrue(np.allclose(x.numpy(), z.numpy(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(
        self, m, k, n, trans_a, trans_b, fuse_add_to_output, dtype=np.float32
    ):
        shape_a = self.make_shape(m, k, trans_a)
        shape_b = self.make_shape(k, n, trans_b)
        a = np.ones(shape_a, dtype=dtype)
        b = np.ones(shape_b, dtype=dtype)
        self._test_body(a, b, trans_a, trans_b, fuse_add_to_output, dtype=dtype)

    def _test_random_body(
        self, m, k, n, trans_a, trans_b, fuse_add_to_output, dtype=np.float32
    ):
        shape_a = self.make_shape(m, k, trans_a)
        shape_b = self.make_shape(k, n, trans_b)
        a = np.random.random(shape_a).astype(dtype)
        b = np.random.random(shape_b).astype(dtype)
        self._test_body(a, b, trans_a, trans_b, fuse_add_to_output, dtype=dtype)

    def test_ones1x1x1_input(self):
        print("run test_ones1x1x1_input: ")
        self._test_ones_body(1, 1, 1, False, False, True)
        self._test_ones_body(1, 1, 1, False, False, False)
        self._test_ones_body(1, 1, 1, False, True, True)
        self._test_ones_body(1, 1, 1, False, True, False)
        self._test_ones_body(1, 1, 1, True, False, True)
        self._test_ones_body(1, 1, 1, True, False, False)
        self._test_ones_body(1, 1, 1, True, True, True)
        self._test_ones_body(1, 1, 1, True, True, False)

    def test_random1x1x1_input(self):
        print("test_random1x1x1_input: ")
        self._test_random_body(1, 1, 1, False, False, True)
        self._test_random_body(1, 1, 1, False, False, False)
        self._test_random_body(1, 1, 1, False, True, True)
        self._test_random_body(1, 1, 1, False, True, False)
        self._test_random_body(1, 1, 1, True, False, True)
        self._test_random_body(1, 1, 1, True, False, False)
        self._test_random_body(1, 1, 1, True, True, True)
        self._test_random_body(1, 1, 1, True, True, False)


def test_ones1x10x1_input(self):
    print("test_ones1x10x1_input: ")
    self._test_ones_body(1, 10, 1, False, False, True)
    self._test_ones_body(1, 10, 1, False, False, False)
    self._test_ones_body(1, 10, 1, False, True, True)
    self._test_ones_body(1, 10, 1, False, True, False)
    self._test_ones_body(1, 10, 1, True, False, True)
    self._test_ones_body(1, 10, 1, True, False, False)
    self._test_ones_body(1, 10, 1, True, True, True)
    self._test_ones_body(1, 10, 1, True, True, False)


def test_random1x10x1_input(self):
    print("test_random1x10x1_input: ")
    self._test_random_body(1, 10, 1, False, False, True)
    self._test_random_body(1, 10, 1, False, True, True)
    self._test_random_body(1, 10, 1, True, False, True)
    self._test_random_body(1, 10, 1, True, True, True)


def test_ones10x10x2_input(self):
    print("test_ones10x10x2_input: ")
    self._test_ones_body(10, 10, 2, False, False, True)
    self._test_ones_body(10, 10, 2, False, True, True)
    self._test_ones_body(10, 10, 2, True, False, True)
    self._test_ones_body(10, 10, 2, True, True, True)


def test_random10x10x2_input(self):
    print("run test_random10x10x2_input: ")
    self._test_random_body(10, 10, 2, False, False, True)
    self._test_random_body(10, 10, 2, False, True, True)
    self._test_random_body(10, 10, 2, True, False, True)
    self._test_random_body(10, 10, 2, True, True, True)


if __name__ == "__main__":
    unittest.main()
