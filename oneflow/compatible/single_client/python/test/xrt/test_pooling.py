"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest

import numpy as np
import oneflow as flow

config = flow.function_config()


class TestPooling(unittest.TestCase):
    run_test = False

    def _test_body(self, x, ksize, strides, padding, data_format, dtype=np.float32):
        if not self.run_test:
            return
        f1 = self.make_job(
            x.shape, ksize, strides, padding, data_format, dtype=flow.float32
        )
        f2 = self.make_trt_job(
            x.shape, ksize, strides, padding, data_format, dtype=flow.float32
        )
        a = f1(x).get()
        b = f2(x).get()
        print("without trt: ", a)
        print("with tensorrt", b)
        self.assertTrue(a.shape == b.shape)
        self.assertTrue(np.allclose(a.numpy(), b.numpy(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(
        self, shape, ksize, strides, padding, data_format, dtype=np.float32
    ):
        x = np.ones(shape, dtype=dtype)
        self._test_body(
            x,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dtype=dtype,
        )

    def _test_random_body(
        self, shape, ksize, strides, padding, data_format, dtype=np.float32
    ):
        x = np.random.random(shape).astype(dtype)
        self._test_body(
            x,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dtype=dtype,
        )

    def test_ones_input(self):
        print("test ones input: ")
        self._test_ones_body((1, 1, 6, 6), 1, 1, "VALID", "NCHW")
        self._test_ones_body((1, 3, 6, 6), 3, 2, "SAME", "NCHW")
        self._test_ones_body((1, 1, 3, 3), 1, 1, "VALID", "NCHW")
        self._test_ones_body((1, 5, 9, 9), 3, 1, "SAME", "NCHW")
        self._test_ones_body((1, 7, 9, 9), 1, 1, "SAME", "NCHW")
        self._test_ones_body((1, 5, 3, 3), 1, 1, "VALID", "NCHW")
        self._test_ones_body((1, 1, 6, 6), 2, 2, "SAME", "NCHW")
        self._test_ones_body((1, 1, 6, 6), 2, 2, "VALID", "NCHW")
        self._test_ones_body((1, 1, 9, 9), 2, 2, "SAME", "NCHW")
        self._test_ones_body((1, 1, 9, 9), 2, 2, "VALID", "NCHW")

    #   self._test_ones_body((1, 224, 224, 3), 3, 2, "VALID", "NHWC")
    #   self._test_ones_body((1, 224, 224, 1), 2, 1, "SAME", "NHWC")

    def test_random_input(self):
        print("test random input: ")
        self._test_random_body((1, 1, 6, 6), 1, 1, "VALID", "NCHW")
        self._test_random_body((1, 3, 6, 6), 3, 2, "SAME", "NCHW")
        self._test_random_body((1, 5, 6, 6), 3, 2, "VALID", "NCHW")
        self._test_random_body((1, 7, 6, 6), 3, 2, "SAME", "NCHW")
        self._test_random_body((1, 3, 3, 3), 1, 1, "VALID", "NCHW")
        self._test_random_body((1, 3, 6, 6), 3, 2, "SAME", "NCHW")
        self._test_random_body((1, 1, 6, 6), 2, 2, "SAME", "NCHW")
        self._test_random_body((1, 1, 6, 6), 2, 2, "VALID", "NCHW")
        self._test_random_body((1, 1, 9, 9), 2, 2, "SAME", "NCHW")
        self._test_random_body((1, 1, 9, 9), 2, 2, "VALID", "NCHW")

    # self._test_random_body((1, 224, 224, 3), 3, 2, "VALID", "NHWC")
    # self._test_random_body((1, 224, 224, 1), 2, 1, "SAME", "NHWC")


class TestMaxPooling(TestPooling):
    run_test = True

    def make_job(
        self, x_shape, ksize, strides, padding, data_format, dtype=flow.float32
    ):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def max_pooling_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.nn.max_pool2d(
                x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )

        return max_pooling_job

    def make_trt_job(
        self, x_shape, ksize, strides, padding, data_format, dtype=flow.float32
    ):
        config.use_xla_jit(False)
        config.use_tensorrt(True)

        @flow.global_function(config)
        def trt_max_pooling_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.nn.max_pool2d(
                x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )

        return trt_max_pooling_job


class TestAveragePooling(TestPooling):
    run_test = True

    def make_job(
        self, x_shape, ksize, strides, padding, data_format, dtype=flow.float32
    ):
        config.use_xla_jit(False)
        config.use_tensorrt(False)

        @flow.global_function(config)
        def avg_pooling_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.nn.avg_pool2d(
                x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )

        return avg_pooling_job

    def make_trt_job(
        self, x_shape, ksize, strides, padding, data_format, dtype=flow.float32
    ):
        config.use_xla_jit(False)
        config.use_tensorrt(True)

        @flow.global_function(config)
        def trt_avg_pooling_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
            return flow.nn.avg_pool2d(
                x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )

        return trt_avg_pooling_job


if __name__ == "__main__":
    unittest.main()
