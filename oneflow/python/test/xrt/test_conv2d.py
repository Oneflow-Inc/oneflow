import unittest
import numpy as np

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

def _get_bias_initializer():
    bias_initializer = op_conf_util.InitializerConf()
    bias_initializer.constant_conf.value = 0.0
    return bias_initializer

def _conv2d(
    input,
    filters,
    kernel_size,
    strides,
    padding,
    data_format,
    dilation_rate):
    if data_format == "NCHW":
        weight_shape = (filters, input.static_shape[1], kernel_size, kernel_size)
    else:
        weight_shape = (filters, kernel_size, kernel_size, input.static_shape[3])
    weight = flow.get_variable(
        "weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=flow.random_uniform_initializer(),
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate)
    return output

def make_job(a_shape, filters=None, kernel_size=None, strides=None,
        padding="valid", data_format="NCHW", dilation_rate=None, dtype=flow.float32):
    @flow.function
    def conv2d_job(a=flow.input_blob_def(a_shape, dtype=dtype)):
        flow.config.use_xla_jit(False)
        flow.config.use_tensorrt(False)
        return _conv2d(a, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding,
                data_format=data_format,dilation_rate=dilation_rate)
    return conv2d_job

def make_trt_job(a_shape, filters=None, kernel_size=None, strides=None,
        padding="valid", data_format="NCHW", dilation_rate=None, dtype=flow.float32):
    @flow.function
    def trt_conv2d_job(a=flow.input_blob_def(a_shape, dtype=dtype)):
        flow.config.use_xla_jit(False)
        flow.config.use_tensorrt(True)
        return _conv2d(a, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding,
                data_format=data_format,dilation_rate=dilation_rate)
    return trt_conv2d_job

class Testconv2d(unittest.TestCase):
    def make_shape(self, shape=[]):
        assert len(shape) == 4
        return (shape[0], shape[1], shape[2], shape[3])

    def _test_body(self, a, filters, kernel_size, strides, padding, data_format,
            dilation_rate, dtype=np.float32):
        f1 = make_job(a.shape, filters, kernel_size, strides, padding,
                data_format, dilation_rate, dtype=flow.float32)

        f2 = make_trt_job(a.shape, filters, kernel_size, strides, padding,
                data_format, dilation_rate, dtype=flow.float32)

        x = f1(a).get()
        y = f2(a).get()
        print("without xla: ", x)
        print("with tensorrt: ", y)
        self.assertTrue(np.allclose(x, y, rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, filters, kernel_size, strides, padding,
            data_format, dilation_rate, shape=[], dtype=np.float32):
        shape_a = self.make_shape(shape)
        print("shape_a: ", shape_a)
        a = np.ones(shape_a, dtype=dtype)
        self._test_body(a, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding, data_format=data_format,
                dilation_rate=dilation_rate)

    def _test_random_body(self, filters, kernel_size, strides,
            padding, data_format, dilation_rate, shape=[], dtype=np.float32):
        shape_a = self.make_shape(shape)
        print("shape_a", shape_a)
        a = np.random.random(shape_a).astype(dtype)
        self._test_body(a, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding, data_format=data_format,
                dilation_rate=dilation_rate)

    def test_ones1x1_input(self):
        print("run test_ones1x1_input: ")
        self._test_ones_body(shape=[1, 1, 1, 1], filters=1, kernel_size=1, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1)
        self._test_ones_body(shape=[1, 3, 1, 1], filters=1, kernel_size=1, strides=1,
              padding="SAME", data_format="NCHW", dilation_rate=1)
        self._test_ones_body(shape=[1, 5, 1, 1], filters=1, kernel_size=1, strides=1,
              padding="VALID", data_format="NCHW", dilation_rate=1)
     #   self._test_ones_body(shape=[1, 1, 1, 7], filters=1, kernel_size=1,
     # strides=1, padding="SAME", data_format="NHWC", dilation_rate=1)

    def test_random1x1_input(self):
        print("test_random1x1_input: ")
        self._test_random_body(shape=[1, 1, 1, 1], filters=1, kernel_size=1, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1)
        self._test_random_body(shape=[1, 3, 1, 1], filters=1, kernel_size=1, strides=1,
                padding="SAME", data_format="NCHW", dilation_rate=1)
        self._test_random_body(shape=[1, 5, 1, 1], filters=1, kernel_size=1, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1)
    #   self._test_random_body(shape=[1, 1, 1, 7], filters=1, kernel_size=1,
    # strides=1, padding="SAME", data_format="NHWC", dilation_rate=1)

    def test_ones3x3_input(self):
        print("test_ones3x3_input: ")
        self._test_ones_body(shape=[1, 1, 3, 3], filters=3, kernel_size=3, strides=2,
                padding="VALID", data_format="NCHW", dilation_rate=1)
        self._test_ones_body(shape=[1, 3, 3, 3], filters=1, kernel_size=1, strides=1,
                padding="SAME", data_format="NCHW", dilation_rate=1)
        self._test_ones_body(shape=[1, 5, 3, 3], filters=3, kernel_size=1, strides=1,
                 padding="VALID", data_format="NCHW", dilation_rate=1)
     #   self._test_ones_body(shape=[1, 3, 3, 7], filters=3, kernel_size=3,
     # strides=1, padding="SAME", data_format="NHWC", dilation_rate=1)

    def test_random3x3_input(self):
        print("test_random3x3_input: ")
        self._test_random_body(shape=[1, 1, 3, 3], filters=3, kernel_size=3, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1)
        self._test_random_body(shape=[1, 3, 3, 3], filters=3, kernel_size=1,
                strides=1,
                padding="SAME", data_format="NCHW", dilation_rate=1)
      #  self._test_random_body(shape=[1, 3, 3, 1], filters=3, kernel_size=1,
      # strides=1, padding="VALID", data_format="NHWC", dilation_rate=1)
      #  self._test_random_body(shape=[1, 3, 3, 7], filters=1, kernel_size=1, 
      # strides=1, padding="SAME", data_format="NHWC", dilation_rate=1)

    def test_ones227x227_input(self):
        print("test_ones227x227_input: ")
        self._test_ones_body(shape=[1, 3, 227, 227], filters=64, kernel_size=11,
                strides=4, padding="VALID", data_format="NCHW", dilation_rate=1)
        self._test_ones_body(shape=[1, 3, 227, 227], filters=64, kernel_size=11,
                strides=4, padding="SAME", data_format="NCHW", dilation_rate=1)
      #  self._test_ones_body(shape=[1, 227, 227, 3], filters=64,
      #  kernel_size=11, strides=4, padding="VALID", data_format="NHWC",
      #  dilation_rate=1)
      #  self._test_ones_body(shape=[1, 227, 227, 3], filters=64,
      #  kernel_size=11, strides=4,
      #  padding="SAME", data_format="NHWC", dilation_rate=1)

    def test_random227x227_input(self):
        print("run test_random227x227_input: ")
        self._test_random_body(shape=[1, 3, 227, 227], filters=64, kernel_size=11,
                strides=4, padding="VALID", data_format="NCHW", dilation_rate=1)
        self._test_random_body(shape=[1, 3, 227, 227], filters=64, kernel_size=11,
                strides=4, padding="SAME", data_format="NCHW", dilation_rate=1)
     #   self._test_random_body(shape=[1, 227, 227, 3], filters=64, kernel_size=11,
     #           strides=4, padding="VALID", data_format="NHWC", dilation_rate=1)
     #   self._test_random_body(shape=[1, 227, 227, 3], filters=64, kernel_size=11,
     #           strides=4, padding="SAME", data_format="NHWC", dilation_rate=1)

if __name__ == '__main__':
  unittest.main()
