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


def make_job(
    x_shape,
    w_shape,
    strides=None,
    padding="valid",
    bias=None,
    data_format="NCHW",
    dilation_rate=None,
    groups=1,
    xrts=[]
):
    config = flow.FunctionConfig()
    xrt_util.set_xrt(config, xrts=xrts)
    @flow.global_function(function_config=config)
    def conv2d_job(
        x:oft.Numpy.Placeholder(x_shape),
        weight:oft.Numpy.Placeholder(w_shape),
    ):
        return flow.nn.conv2d(
            x, weight, strides, padding, bias, data_format, dilation_rate, groups
        )

    return conv2d_job


class TestConv2d(unittest.TestCase):
    def make_filter_shape(self, shape, filters, kernel_size, data_format):
        if data_format == "NCHW":
            return [filters, shape[1], kernel_size, kernel_size]
        else:
            return [filters, kernel_size, kernel_size, shape[3]]

    def _test_body(
        self,
        x,
        filters,
        strides,
        padding,
        data_format,
        dilation_rate,
    ):
        func = make_job(
            x.shape,
            filters.shape,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate)

        out = func(x, filters).get()
        out_np = out.numpy()
        flow.clear_default_session()
        for xrt in xrt_util.xrt_backends:
            xrt_job = make_job(
                x.shape,
                filters.shape,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate)
            xrt_out = xrt_job(x, filters).get()
            self.assertTrue(np.allclose(out_np, xrt_out.numpy(), rtol=0.001, atol=1e-05))
            flow.clear_default_session()

    def _test_ones_body(
        self,
        shape,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        dtype=np.float32,
    ):
        assert len(shape) == 4
        x = np.ones(shape, dtype=dtype)
        w_shape = self.make_filter_shape(shape, filters, kernel_size, data_format)
        weight = np.random.random(w_shape).astype(dtype)
        self._test_body(
            x,
            weight,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

    def _test_random_body(
        self,
        shape,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dilation_rate,
        dtype=np.float32,
    ):
        assert len(shape) == 4
        x = np.random.random(shape).astype(dtype)
        w_shape = self.make_filter_shape(shape, filters, kernel_size, data_format)
        weight = np.random.random(w_shape).astype(dtype)
        self._test_body(
            x,
            weight,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

    def test_ones_kernel_1x1(self):
        self._test_ones_body(
            shape=[1, 1, 1, 1],
            filters=1,
            kernel_size=1,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_ones_body(
            shape=[1, 3, 1, 1],
            filters=1,
            kernel_size=1,
            strides=1,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_ones_body(
            shape=[1, 1, 5, 5],
            filters=1,
            kernel_size=1,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_ones_body(
            shape=[3, 3, 5, 5],
            filters=1,
            kernel_size=1,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )

    def test_random_kernel_1x1(self):
        self._test_random_body(
            shape=[1, 1, 1, 1],
            filters=1,
            kernel_size=1,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_random_body(
            shape=[1, 3, 1, 1],
            filters=1,
            kernel_size=1,
            strides=1,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_random_body(
            shape=[1, 1, 5, 5],
            filters=1,
            kernel_size=1,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_random_body(
            shape=[3, 3, 5, 5],
            filters=1,
            kernel_size=1,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )

    def test_ones_kernel_3x3(self):
        self._test_ones_body(
            shape=[1, 1, 3, 3],
            filters=1,
            kernel_size=3,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_ones_body(
            shape=[1, 3, 5, 5],
            filters=1,
            kernel_size=3,
            strides=1,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_ones_body(
            shape=[1, 5, 3, 3],
            filters=1,
            kernel_size=3,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )

    def test_random_kernel_3x3(self):
        self._test_random_body(
            shape=[1, 1, 3, 3],
            filters=1,
            kernel_size=3,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_random_body(
            shape=[1, 3, 3, 3],
            filters=1,
            kernel_size=3,
            strides=1,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_random_body(
            shape=[1, 3, 3, 3],
            filters=1,
            kernel_size=3,
            strides=1,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_random_body(
            shape=[1, 3, 3, 3],
            filters=1,
            kernel_size=3,
            strides=1,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )

    def test_ones_kernel_11x11(self):
        self._test_ones_body(
            shape=[1, 3, 24, 24],
            filters=3,
            kernel_size=11,
            strides=4,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_ones_body(
            shape=[1, 3, 24, 24],
            filters=3,
            kernel_size=11,
            strides=4,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_ones_body(
            shape=[1, 3, 27, 27],
            filters=3,
            kernel_size=11,
            strides=4,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_ones_body(
            shape=[1, 3, 27, 27],
            filters=3,
            kernel_size=11,
            strides=4,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )

    def test_random_kernel_11x11(self):
        self._test_random_body(
            shape=[1, 3, 24, 24],
            filters=3,
            kernel_size=11,
            strides=4,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_random_body(
            shape=[1, 3, 24, 24],
            filters=3,
            kernel_size=11,
            strides=4,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_random_body(
            shape=[1, 3, 27, 27],
            filters=3,
            kernel_size=11,
            strides=4,
            padding="VALID",
            data_format="NCHW",
            dilation_rate=1,
        )
        self._test_random_body(
            shape=[1, 3, 27, 27],
            filters=3,
            kernel_size=11,
            strides=4,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
        )


if __name__ == "__main__":
    unittest.main()
