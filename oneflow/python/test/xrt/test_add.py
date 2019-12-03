import unittest
import numpy as np

import oneflow as flow

@flow.function
def add_job(x=flow.input_blob_def((3, 3)), y=flow.input_blob_def((3, 3))):
    flow.config.use_xla_jit(False)
    flow.config.use_tensorrt(False)
    flow.config.default_data_type(flow.float)
    
    return flow.math.add(x, y)

@flow.function
def trt_add_job(x=flow.input_blob_def((3, 3)), y=flow.input_blob_def((3, 3))):
    flow.config.use_xla_jit(False)
    flow.config.use_tensorrt(True)
    flow.config.default_data_type(flow.float)

    return flow.math.add(x, y)

# @flow.function
# def trt_add_job(x = flow.input_blob_def((10,))):
#     flow.config.use_xla_jit(False)
#     flow.config.use_tensorrt(True)
#     return flow.keras.activations.add(x)


class Testadd(unittest.TestCase):
    def _test_body(self, x, y):
        a = add_job(x, y).get()
        b = trt_add_job(x, y).get()
        print("without tensorrt: ", a)
        print("with tensorrt", b)
        self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))
        # b = trt_add_job(x).get()
        # print("with tensorrt", b)
        # self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))

    def test_ones_input(self):
        x = np.ones((3, 3), dtype=np.float32)
        y = np.ones((3, 3), dtype=np.float32)
        self._test_body(x, y)

    def test_random_input(self):
        x = np.random.rand(3, 3).astype(np.float32)
        y = np.random.rand(3, 3).astype(np.float32)
        self._test_body(x, y)



if __name__ == '__main__':
  unittest.main()

