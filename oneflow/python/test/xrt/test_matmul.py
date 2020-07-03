import unittest

import numpy as np
import oneflow as flow

config = flow.function_config()


def make_job(a_shape, b_shape, trans_a=False, trans_b=False, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def matmul_job(
        a=flow.FixedTensorDef(a_shape, dtype=dtype),
        b=flow.FixedTensorDef(b_shape, dtype=dtype),
    ):
        return flow.matmul(a, b, transpose_a=trans_a, transpose_b=trans_b)

    return matmul_job


def make_xla_job(a_shape, b_shape, trans_a=False, trans_b=False, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def xla_matmul_job(
        a=flow.FixedTensorDef(a_shape, dtype=dtype),
        b=flow.FixedTensorDef(b_shape, dtype=dtype),
    ):
        return flow.matmul(a, b, transpose_a=trans_a, transpose_b=trans_b)

    return xla_matmul_job


class TestMatmul(unittest.TestCase):
    def make_shape(self, m, n, transpose):
        if transpose:
            return (n, m)
        else:
            return (m, n)

    def _test_body(self, a, b, trans_a, trans_b, dtype=np.float32):
        f1 = make_job(a.shape, b.shape, trans_a, trans_b)
        f2 = make_xla_job(a.shape, b.shape, trans_a, trans_b)
        x = f1(a, b).get()
        y = f2(a, b).get()
        print("without xla: ", x)
        print("with xla: ", y)
        self.assertTrue(np.allclose(x.ndarray(), y.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, m, k, n, trans_a, trans_b, dtype=np.float32):
        shape_a = self.make_shape(m, k, trans_a)
        shape_b = self.make_shape(k, n, trans_b)
        a = np.ones(shape_a, dtype=dtype)
        b = np.ones(shape_b, dtype=dtype)
        self._test_body(a, b, trans_a, trans_b, dtype=dtype)

    def _test_random_body(self, m, k, n, trans_a, trans_b, dtype=np.float32):
        shape_a = self.make_shape(m, k, trans_a)
        shape_b = self.make_shape(k, n, trans_b)
        a = np.random.random(shape_a).astype(dtype)
        b = np.random.random(shape_b).astype(dtype)
        self._test_body(a, b, trans_a, trans_b, dtype=dtype)

    def test_ones1x1x1_input(self):
        print("run test_ones1x1x1_input: ")
        self._test_ones_body(1, 1, 1, False, False)
        self._test_ones_body(1, 1, 1, False, True)
        self._test_ones_body(1, 1, 1, True, False)
        self._test_ones_body(1, 1, 1, True, True)

    def test_random1x1x1_input(self):
        print("test_random1x1x1_input: ")
        self._test_random_body(1, 1, 1, False, False)
        self._test_random_body(1, 1, 1, False, True)
        self._test_random_body(1, 1, 1, True, False)
        self._test_random_body(1, 1, 1, True, True)

    def test_ones1x10x1_input(self):
        print("test_ones1x10x1_input: ")
        self._test_ones_body(1, 10, 1, False, False)
        self._test_ones_body(1, 10, 1, False, True)
        self._test_ones_body(1, 10, 1, True, False)
        self._test_ones_body(1, 10, 1, True, True)

    def test_random1x10x1_input(self):
        print("test_random1x10x1_input: ")
        self._test_random_body(1, 10, 1, False, False)
        self._test_random_body(1, 10, 1, False, True)
        self._test_random_body(1, 10, 1, True, False)
        self._test_random_body(1, 10, 1, True, True)

    def test_ones10x10x2_input(self):
        print("test_ones10x10x2_input: ")
        self._test_ones_body(10, 10, 2, False, False)
        self._test_ones_body(10, 10, 2, False, True)
        self._test_ones_body(10, 10, 2, True, False)
        self._test_ones_body(10, 10, 2, True, True)

    def test_random10x10x2_input(self):
        print("run test_random10x10x2_input: ")
        self._test_random_body(10, 10, 2, False, False)
        self._test_random_body(10, 10, 2, False, True)
        self._test_random_body(10, 10, 2, True, False)
        self._test_random_body(10, 10, 2, True, True)


if __name__ == "__main__":
    unittest.main()
