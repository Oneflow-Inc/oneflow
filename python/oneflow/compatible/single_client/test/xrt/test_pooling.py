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

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
import oneflow.compatible.single_client.typing as oft

import xrt_util


class TestPooling(unittest.TestCase):
    run_test = False
    def _test_body(self, x, ksize, strides, padding, data_format):
        if not self.run_test:
            return

        func = self.make_job(
            x.shape, ksize, strides, padding, data_format
        )
        out = func(x).get()
        out_np = out.numpy()
        flow.clear_default_session()

        for xrt in xrt_util.xrt_backends:
            xrt_job = self.make_job(
                x.shape, ksize, strides, padding, data_format, xrts=[xrt]
            )
            xrt_out = xrt_job(x).get()
            self.assertTrue(np.allclose(out_np, xrt_out.numpy(), rtol=0.001, atol=1e-05))
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


class TestMaxPooling(TestPooling):
    run_test = True

    def make_job(
        self, x_shape, ksize, strides, padding, data_format, xrts=[]
    ):
        config = flow.FunctionConfig()
        xrt_util.set_xrt(config, xrts=xrts)

        @flow.global_function(function_config=config)
        def max_pooling_job(x:oft.Numpy.Placeholder(x_shape)):
            return flow.nn.max_pool2d(
                x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )

        return max_pooling_job

class TestAveragePooling(TestPooling):
    run_test = True

    def make_job(
        self, x_shape, ksize, strides, padding, data_format, xrts=[]
    ):
        config = flow.FunctionConfig()
        xrt_util.set_xrt(config, xrts=xrts)

        @flow.global_function(function_config=config)
        def avg_pooling_job(x:oft.Numpy.Placeholder(x_shape)):
            return flow.nn.avg_pool2d(
                x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )

        return avg_pooling_job

if __name__ == "__main__":
    unittest.main()
