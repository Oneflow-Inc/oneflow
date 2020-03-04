import unittest
import numpy as np

import oneflow as flow

config = flow.function_config()

class TestGather(unittest.TestCase):
    def _test_body(self, x, indices, axis, dtype=flow.float32):
        indices = np.array(indices).astype(np.int32)
        f1 = self.make_job(x.shape, indices.shape, axis, dtype=dtype)
        f2 = self.make_xla_job(x.shape, indices.shape, axis, dtype=dtype)
        a = f1(x, indices).get()
        b = f2(x, indices).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def make_job(self, input_shape, indices_shape, axis, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.function(config)
        def gather_job(x = flow.FixedTensorDef(input_shape, dtype=dtype),
                       indices = flow.FixedTensorDef(indices_shape, dtype=flow.int32)):
            return flow.gather(x, indices, axis=axis)
        return gather_job

    def make_xla_job(self, input_shape, indices_shape, axis, dtype=flow.float32):
        config.use_xla_jit(True)
        config.use_tensorrt(False)

        @flow.function(config)
        def xla_gather_job(x = flow.FixedTensorDef(input_shape, dtype=dtype),
                           indices = flow.FixedTensorDef(indices_shape, dtype=flow.int32)):
            return flow.gather(x, indices, axis=axis)
        return xla_gather_job

    def _test_ones_body(self, shape, indices, axis, dtype=flow.float32):
        np_dtype = flow.convert_of_dtype_to_numpy_dtype(dtype)
        x = np.ones(shape, dtype=np_dtype)
        self._test_body(x, indices, axis, dtype=dtype)

    def _test_random_body(self, shape, indices, axis, dtype=flow.float32):
        np_dtype = flow.convert_of_dtype_to_numpy_dtype(dtype)
        x = np.random.random(shape).astype(np_dtype)
        self._test_body(x, indices, axis, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1, 1), [0], 0)
        self._test_ones_body((2, 2), [0, 0], 0)
        self._test_ones_body((1, 10), [[0], [0]], 0)
        self._test_ones_body((1, 10), [[0, 1, 2], [2, 3, 4]], 1)
        self._test_ones_body((2, 10, 2), [[0, 1], [2, 3], [4, 5]], 1)
        self._test_ones_body((2, 5, 2, 2), [[0, 0], [1, 1]], 3)

    def test_random_input(self):
        self._test_random_body((1, 1), [0], 0)
        self._test_random_body((2, 2), [0, 0], 0)
        self._test_random_body((1, 10), [[0], [0]], 0)
        self._test_random_body((1, 10), [[0, 1, 2], [2, 3, 4]], 1)
        self._test_random_body((2, 10, 2), [[0, 1], [2, 3], [4, 5]], 1)
        self._test_random_body((2, 5, 2, 2), [[0, 0], [1, 1]], 3)

class TestBatchGather(TestGather):
    def make_job(self, input_shape, indices_shape, axis, dtype=flow.float32):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.function(config)
        def batch_gather_job(x = flow.FixedTensorDef(input_shape, dtype=dtype),
                             indices = flow.FixedTensorDef(indices_shape,
                             dtype=flow.int32)):
            return flow.gather(x, indices, batch_dims=axis)
        return batch_gather_job

    def make_xla_job(self, input_shape, indices_shape, axis, dtype=flow.float32):
        config.use_xla_jit(True)
        config.use_tensorrt(False)

        @flow.function(config)
        def xla_batch_gather_job(x = flow.FixedTensorDef(input_shape, dtype=dtype),
                                 indices = flow.FixedTensorDef(indices_shape, dtype=flow.int32)):
            return flow.gather(x, indices, batch_dims=axis)
        return xla_batch_gather_job

    def test_ones_input(self):
        # batch_dims should be Dims(indices) - 1 and batch_dims > 0
        self._test_ones_body((2, 3, 2), [[0], [1]], 1)
        self._test_ones_body((2, 3, 2), [[0, 1], [1, 0]], 1)
        self._test_ones_body((2, 3, 2, 2), [[[0], [0], [0]], [[1], [1], [1]]], 2)

    def test_random_input(self):
        # batch_dims should be Dims(indices) - 1 and batch_dims > 0
        self._test_random_body((2, 3, 2), [[0], [1]], 1)
        self._test_random_body((2, 3, 2), [[0, 1], [1, 2]], 1)
        self._test_random_body((2, 3, 2, 2), [[[0], [0], [0]], [[1], [1], [1]]], 2)

if __name__ == '__main__':
    unittest.main()
