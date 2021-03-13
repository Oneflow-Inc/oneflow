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
import oneflow as flow
import unittest
import numpy as np
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)


class NumpyAvgPooling2D:
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        self.kernel_size = (
            _pair(kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = _pair(stride) if isinstance(stride, int) else stride
        self.padding = _pair(padding) if isinstance(padding, int) else padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def _to_int(self, x):
        if self.ceil_mode:
            return int(np.ceil(x))
        else:
            return int(np.floor(x))

    def _avg_pool2d_on_mat(self, mat):
        h_in = mat.shape[0]
        w_in = mat.shape[1]
        h_out = self._to_int(
            ((h_in + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1
        )
        w_out = self._to_int(
            ((w_in + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1
        )
        h_stride = self.stride[0]
        w_stride = self.stride[1]
        h_kernel = self.kernel_size[0]
        w_kernel = self.kernel_size[1]

        out = np.zeros((h_out, w_out))

        def _mean(start_row, end_row, start_col, end_col):
            sum = 0
            for row in range(start_row, end_row):
                for col in range(start_col, end_col):
                    sum = sum + mat[row][col]
            return sum / (h_kernel * w_kernel)

        for row in range(h_out):
            for col in range(w_out):
                start_row = row * h_stride
                start_col = col * w_stride
                end_row = start_row + h_kernel
                end_col = start_col + w_kernel
                out[row][col] = _mean(start_row, end_row, start_col, end_col)
        return out

    def __call__(self, x):
        # x: nchw
        self.x = x
        out = []
        for sample in x:
            channels = []
            for mat in sample:
                channels.append(self._avg_pool2d_on_mat(mat))
            out.append(channels)
        return np.array(out)


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_AvgPool2d(test_case):
        of_avgpool2d = flow.nn.AvgPool2d(2, stride=1)
        np_avgpool2d = NumpyAvgPooling2D(2, stride=1)
        x = flow.Tensor(np.random.rand(1, 1, 5, 5))
        of_y = of_avgpool2d(x)
        np_y = np_avgpool2d(x.numpy())
        assert np.allclose(of_y.numpy(), np_y)


if __name__ == "__main__":
    unittest.main()
