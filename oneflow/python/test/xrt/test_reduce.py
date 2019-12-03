import unittest
import numpy as np

import oneflow as flow

def MakeReduceSumJob(input_tensor_shape, axis=None, keep_dims=False):
    @flow.function
    def reduce_sum_job(input_tensor=flow.input_blob_def(input_tensor_shape)):
        flow.config.use_xla_jit(False)
        flow.config.use_tensorrt(False)
        flow.config.default_data_type(flow.float)
        return flow.math.reduce_sum(input_tensor,
                                    axis=axis,
                                    keepdims=keep_dims)
    return reduce_sum_job

def TrtMakeReduceSumJob(input_tensor_shape, axis=None, keep_dims=False):
    @flow.function
    def trt_reduce_sum_job(input_tensor=flow.input_blob_def(input_tensor_shape)):
        flow.config.use_xla_jit(False)
        flow.config.use_tensorrt(True)
        flow.config.default_data_type(flow.float)
        return flow.math.reduce_sum(input_tensor,
                                    axis=axis,
                                    keepdims=keep_dims)
    return trt_reduce_sum_job

class Testreduce_sum(unittest.TestCase):
   # def setUp(self):
   #     flow.clear_default_session()

    def _test_body(self, input_shape, axis, keep_dims):
        input_tensor = np.random.random_sample(input_shape).astype(np.float32)
        print("type input_tensor: ", type(input_tensor))
        reduce_sum_job = MakeReduceSumJob(input_tensor_shape=input_shape, axis=axis, keep_dims=keep_dims);
        trt_reduce_sum_job = TrtMakeReduceSumJob(input_tensor_shape=input_shape, axis=axis, keep_dims=keep_dims);
 
        a = reduce_sum_job(input_tensor).get()
        b = trt_reduce_sum_job(input_tensor).get()
        print("without tensorrt: ", a)
        print("with tensorrt", b)
        self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))

    def test_random_input(self):
        input_shape=(128,128,128)
        self._test_body(input_shape=input_shape, axis=(0,2), keep_dims=False)


if __name__ == '__main__':
  unittest.main()

