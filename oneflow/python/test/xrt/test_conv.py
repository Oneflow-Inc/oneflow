import unittest
import numpy as np

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

def _get_bias_initializer():
    bias_initializer = op_conf_util.InitializerConf()
    bias_initializer.constant_conf.value = 0.0
    return bias_initializer

def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=None,
    strides=None,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=None,
    ):
    if data_format == "NCHW":
        weight_shape = (filters, input.static_shape[1], kernel_size, kernel_size)
    else:
        weight_shape = (filters, kernel_size, kernel_size, input.static_shape[3])
    #print("filters: ", filters)
    #print("input shape[1]: ", input.static_shape[1])
    #print("kernel_size: ", kernel_size)
    weight = flow.get_variable(
        name+"-Weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name+"-Bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError
    return output

def make_job(a_shape, filters=None, kernel_size=None, strides=None, 
        padding="valid", data_format="NCHW", dilation_rate=None, use_bias=False,
        weight_initializer=flow.random_uniform_initializer(), bias_initializer=None, dtype=flow.float32):
    @flow.function
    def conv2d_job(a=flow.input_blob_def(a_shape, dtype=dtype)):
        flow.config.use_xla_jit(False)
        flow.config.use_tensorrt(False)
        return _conv2d_layer("conv1", a, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding,
                data_format=data_format,dilation_rate=dilation_rate,
                use_bias=use_bias,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer)
    return conv2d_job

def make_trt_job(a_shape, filters=None, kernel_size=None, strides=None, 
        padding="valid", data_format="NCHW", dilation_rate=None, use_bias=False,
        weight_initializer=flow.random_uniform_initializer(), bias_initializer=None, dtype=flow.float32):
    @flow.function
    def trt_conv2d_job(a=flow.input_blob_def(a_shape, dtype=dtype)):
        flow.config.use_xla_jit(False)
        flow.config.use_tensorrt(True)
        return _conv2d_layer("conv2", a, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding,
                data_format=data_format,dilation_rate=dilation_rate,
                use_bias=use_bias,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer)
    return trt_conv2d_job

class Testconv2d(unittest.TestCase):
    def make_shape(self, m, n, p, q):
        return (m, n, p, q)

    def _test_body(self, a, filters, kernel_size, strides, padding, data_format,
            dilation_rate, use_bias, weight_initializer, bias_initializer,
            dtype=np.float32):
        f1 = make_job(a.shape, filters, kernel_size, strides, padding,
                data_format, dilation_rate, use_bias, weight_initializer,
                bias_initializer)

        f2 = make_trt_job(a.shape, filters, kernel_size, strides, padding,
                data_format, dilation_rate, use_bias, weight_initializer,
                bias_initializer)

        x = f1(a).get()
        y = f2(a).get()
        print("with xla: ", x)
        print("with tensorrt: ", y)
        self.assertTrue(np.allclose(x, y, rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, m, n, p, q, filters, kernel_size, strides, padding, data_format,
            dilation_rate, use_bias, weight_initializer, bias_initializer, dtype=np.float32):
        shape_a = self.make_shape(m, n, p, q)
        a = np.ones(shape_a, dtype=dtype)
        self._test_body(a, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, use_bias=use_bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer, dtype=dtype)

    def _test_random_body(self, m, n, p, q, filters, kernel_size, strides, padding, data_format,
            dilation_rate, use_bias, weight_initializer, bias_initializer, dtype=np.float32):
        shape_a = self.make_shape(m, n,p ,q)
        a = np.random.random(shape_a).astype(dtype)
        self._test_body(a, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, use_bias=use_bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer, dtype=dtype)

    def test_ones1x1_input(self):
        print("run test_ones1x1x1_input: ")
        self._test_ones_body(1, 1, 1, 1, filters=1, kernel_size=1, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=True, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
        self._test_ones_body(1, 3, 1, 1, filters=1, kernel_size=1, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=False, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
        self._test_ones_body(1, 7, 1, 1, filters=1, kernel_size=1, strides=1,
                padding="SAME", data_format="NCHW", dilation_rate=1,
                use_bias=True, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
      #  self._test_ones_body(1, 1, 1, 1, filters=1, kernel_size=1, strides=1,
      #          padding="SAME", data_format="NHWC", dilation_rate=1,
      #          use_bias=False, weight_initializer=flow.random_uniform_initializer(),
      #          bias_initializer=_get_bias_initializer())

    def test_random1x1_input(self):
        print("test_random1x1x1_input: ")
        self._test_random_body(1, 1, 1, 1, filters=1, kernel_size=1, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=True, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
        self._test_random_body(1, 3, 1, 1, filters=1, kernel_size=1, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=False, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
        self._test_random_body(1, 7, 1, 1, filters=1, kernel_size=1, strides=1,
                padding="SAME", data_format="NCHW", dilation_rate=1,
                use_bias=True, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
      #  self._test_random_body(1, 1, 1, 1, filters=1, kernel_size=1, strides=1,
      #          padding="SAME", data_format="NHWC", dilation_rate=1,
      #          use_bias=False, weight_initializer=flow.random_uniform_initializer(),
      #          bias_initializer=_get_bias_initializer())

    def test_ones3x3_input(self):
        print("test_ones3x3_input: ")
        self._test_ones_body(1, 1, 3, 3, filters=2, kernel_size=1, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=True, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
        self._test_ones_body(1,3, 3, 3, filters=2, kernel_size=2, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=False,
                weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=None)
     #   self._test_ones_body(1, 3, 3, 1, filters=1, kernel_size=1, strides=1,
     #          padding="SAME", data_format="NHWC", dilation_rate=1,
     #          use_bias=True,  weight_initializer=flow.random_uniform_initializer(),
     #          bias_initializer=_get_bias_initializer())
     #   self._test_ones_body(1, 3, 3, 3, filters=1, kernel_size=1, strides=1,
     #          padding="VALID", data_format="NHWC", dilation_rate=1,
     #          use_bias=False,
     #          weight_initializer=flow.random_uniform_initializer(),
     #          bias_initializer=None)

    def test_random3x3_input(self):
        print("test_random3x3_input: ")
        self._test_random_body(1, 1, 3, 3, filters=2, kernel_size=1, strides=1,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=True, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
   #     self._test_random_body(1, 3, 3, 3, filters=2, kernel_size=1,
   #             strides=1,
   #             padding="VALID", data_format="NHWC", dilation_rate=1,
   #             use_bias=False,
   #             weight_initializer=flow.random_uniform_initializer(),
   #             bias_initializer=_get_bias_initializer())
   #     self._test_random_body(1, 3, 3, 1, filters=1, kernel_size=1, strides=1,
   #             padding="SAME", data_format="NHWC", dilation_rate=1,
   #             use_bias=True, weight_initializer=flow.random_uniform_initializer(),
   #             bias_initializer=_get_bias_initializer())
   #     self._test_random_body(1, 3, 3, 3, filters=1, kernel_size=1, strides=1,
   #             padding="VALID", data_format="NHWC", dilation_rate=1,
   #             use_bias=False,
   #             weight_initializer=flow.random_uniform_initializer(),
   #             bias_initializer=None)

    def test_ones227x227_input(self):
        print("test_ones227x227_input: ")
        self._test_ones_body(1, 3, 227, 227, filters=64, kernel_size=11,
                strides=4,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=True, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
        self._test_ones_body(1, 3, 227, 227, filters=64, kernel_size=11,
                strides=4,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=False, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
     #   self._test_ones_body(1, 227, 227, 3, filters=64, kernel_size=11,
     #          strides=4,
     #           padding="SAME", data_format="NHWC", dilation_rate=1,
     #           use_bias=True, weight_initializer=flow.random_uniform_initializer(),
     #           bias_initializer=_get_bias_initializer())
     #   self._test_ones_body(1, 227, 227, 3, filters=64, kernel_size=11,
     #           strides=4,
     #           padding="SAME", data_format="NHWC", dilation_rate=1,
     #           use_bias=False, weight_initializer=flow.random_uniform_initializer(),
     #           bias_initializer=None)

    def test_random227x227_input(self):
        print("run test_random227x227_input: ")
        self._test_random_body(1, 3, 227, 227, filters=64, kernel_size=11,
                strides=4,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=True, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
        self._test_random_body(1, 3, 227, 227, filters=64, kernel_size=11,
                strides=4,
                padding="VALID", data_format="NCHW", dilation_rate=1,
                use_bias=False, weight_initializer=flow.random_uniform_initializer(),
                bias_initializer=_get_bias_initializer())
     #   self._test_random_body(1, 227, 227, 3, filters=64, kernel_size=11,
     #           strides=4,
     #           padding="SAME", data_format="NHWC", dilation_rate=1,
     #           use_bias=True, weight_initializer=flow.random_uniform_initializer(),
     #           bias_initializer=_get_bias_initializer())
     #   self._test_random_body(1, 227, 227, 3, filters=64, kernel_size=11,
     #           strides=4,
     #           padding="SAME", data_format="NHWC", dilation_rate=1,
     #           use_bias=False,
     #           weight_initializer=flow.random_uniform_initializer(),
     #           bias_initializer=_get_bias_initializer())

if __name__ == '__main__':
  unittest.main()
