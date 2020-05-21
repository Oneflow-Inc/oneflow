import unittest
import numpy as np

import oneflow as flow

config = flow.function_config()

def make_job(input_shape, dtype=flow.float32, target_dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.function(config)
    def cast_job(x = flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.cast(x, dtype=target_dtype)
    return cast_job

def make_xla_job(input_shape, dtype=flow.float32, target_dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.function(config)
    def xla_cast_job(x = flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.cast(x, dtype=target_dtype)
    return xla_cast_job

class TestCast(unittest.TestCase):
    def _test_body(self, x, dtype=flow.float32, target_dtype=flow.float32):
        f1 = make_job(x.shape, dtype=dtype, target_dtype=target_dtype)
        f2 = make_xla_job(x.shape, dtype=dtype, target_dtype=target_dtype)
        a = f1(x).get()
        b = f2(x).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        # b = trt_cast_job(x).get()
        # print("with tensorrt", b)
        # self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, dtype=flow.float32, target_dtype=flow.float32):
        np_dtype = flow.convert_of_dtype_to_numpy_dtype(dtype)
        x = np.ones(shape, dtype=np_dtype)
        self._test_body(x, dtype=dtype, target_dtype=target_dtype)

    def _test_random_body(self, shape, dtype=flow.float32, target_dtype=flow.float32):
        np_dtype = flow.convert_of_dtype_to_numpy_dtype(dtype)
        x = (1000 * np.random.random(shape)).astype(np_dtype)
        self._test_body(x, dtype=dtype, target_dtype=target_dtype)

    def test_ones_input(self):
        self._test_ones_body((1), flow.float32, flow.int32)
        self._test_ones_body((1, 10), flow.int32, flow.float32)

    def test_random_input(self):
        self._test_random_body((1), flow.float32, flow.int32)
        self._test_random_body((1, 10), flow.int32, flow.float32)

if __name__ == '__main__':
    unittest.main()
