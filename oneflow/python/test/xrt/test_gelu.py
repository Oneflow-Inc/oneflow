import unittest
import numpy as np

import oneflow as flow

@flow.function
def gelu_job(x = flow.input_blob_def((10,))):
    flow.config.use_xla_jit(False)
    return flow.keras.activations.gelu(x)

@flow.function
def xla_gelu_job(x = flow.input_blob_def((10,))):
    flow.config.use_xla_jit(True)
    return flow.keras.activations.gelu(x)

#@flow.function
#def gelu_job_float16(x = flow.input_blob_def((10,), dtype=flow.float16)):
#    flow.config.use_xla_jit(False)
#    #cast = flow.cast(x, dtype= kFloat16)
#    return flow.keras.activations.gelu(x)
#
#@flow.function
#def xla_gelu_job_float16(x = flow.input_blob_def((10,), dtype=flow.float16)):
#    flow.config.use_xla_jit(True)
#    return flow.keras.activations.gelu(x)

class TestGelu(unittest.TestCase):
    def _test_body(self, x):
        a = gelu_job(x).get()
        b = xla_gelu_job(x).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))

    #def _test_body_float16(self, x):
    #    a = gelu_job_float16(x).get()
    #    b = xla_gelu_job_float16(x).get()
    #    print("without xla: ", a)
    #    print("with xla", b)
    #    self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))
     
    def test_float_ones_input(self):
        x = np.ones((10,), dtype=np.float32)
        self._test_body(x)

    def test_float_random_input(self):
        x = np.random.rand(10,).astype(np.float32)
        self._test_body(x)

    #def test_float16_ones_input(self):
    #  x = np.ones((10,), dtype=np.float16)
    #  self._test_body_float16(x)

if __name__ == '__main__':
  unittest.main()
