import unittest
import numpy as np

import oneflow as flow

@flow.function
def relu_job(x = flow.input_blob_def((10,))):
    flow.config.use_xla_jit(False)
    flow.config.use_tensorrt(False)
    return flow.keras.activations.relu(x)

@flow.function
def xla_relu_job(x = flow.input_blob_def((10,))):
    flow.config.use_xla_jit(True)
    flow.config.use_tensorrt(False)
    return flow.keras.activations.relu(x)

# @flow.function
# def trt_relu_job(x = flow.input_blob_def((10,))):
#     flow.config.use_xla_jit(False)
#     flow.config.use_tensorrt(True)
#     return flow.keras.activations.relu(x)

@flow.function
def relu_job_float16(x = flow.input_blob_def((10,), dtype=flow.float16)):
    flow.config.use_xla_jit(False)
    flow.config.use_tensorrt(False)
    return flow.keras.activations.relu(x)

@flow.function
def xla_relu_job_float16(x = flow.input_blob_def((10,), dtype=flow.float16)):
    flow.config.use_xla_jit(True)
    flow.config.use_tensorrt(False)
    return flow.keras.activations.relu(x)

class TestRelu(unittest.TestCase):
    def _test_body(self, x):
        a = relu_job(x).get()
        b = xla_relu_job(x).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))
        # b = trt_relu_job(x).get()
        # print("with tensorrt", b)
        # self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))

    def test_ones_input(self):
        x = np.ones((10,), dtype=np.float32)
        self._test_body(x)

    def test_random_input(self):
        x = np.random.rand(10,).astype(np.float32)
        self._test_body(x)

# class TestReluFloat16(unittest.TestCase):
#     def _test_body(self, x):
#         a = relu_job_float16(x).get()
#         b = xla_relu_job_float16(x).get()
#         print("without xla: ", a)
#         print("with xla", b)
#         self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))
# 
#     def test_ones_input(self):
#         x = np.ones((10,), dtype=np.float16)
#         self._test_body(x)
# 
#     def test_random_input(self):
#         x = np.random.rand(10,).astype(np.float16)
#         self._test_body(x)


if __name__ == '__main__':
  unittest.main()

