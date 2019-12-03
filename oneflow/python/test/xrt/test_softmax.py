import unittest
import numpy as np

import oneflow as flow

@flow.function
def softmax_job(x=flow.input_blob_def((5, 2))):
    flow.config.use_xla_jit(False)
    flow.config.use_tensorrt(False)
    flow.config.default_data_type(flow.float)
    
    return flow.nn.softmax(x, axis=-1)

@flow.function
def trt_softmax_job(x=flow.input_blob_def((5, 2))):
    flow.config.use_xla_jit(False)
    flow.config.use_tensorrt(True)
    flow.config.default_data_type(flow.float)

    return flow.nn.softmax(x, axis=-1)

class Testsoftmax(unittest.TestCase):
    def _test_body(self, x):
        a = softmax_job(x).get()
        b = trt_softmax_job(x).get()
        print("without tensorrt: ", a)
        print("with tensorrt", b)
        self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))
        # b = trt_softmax_job(x).get()
        # print("with tensorrt", b)
        # self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))

    def test_ones_input(self):
        x = np.ones((5, 2), dtype=np.float32)
        y = np.ones((5, 3), dtype=np.float32)
        self._test_body(x)

    def test_random_input(self):
        x = np.random.rand(5, 2).astype(np.float32)
        y = np.random.rand(5, 3).astype(np.float32)
        self._test_body(x)



if __name__ == '__main__':
  unittest.main()

