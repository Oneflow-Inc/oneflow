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
import math
import oneflow.experimental as flow


class MaxPool2dNumpy:
    def __init__(self, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]
        self.min_val = np.finfo(np.float64).min

    def __call__(self, x):
        self.x = x
        self.in_batch = np.shape(x)[0]
        self.in_channel = np.shape(x)[1]
        self.in_height = np.shape(x)[2]
        self.in_width = np.shape(x)[3]

        pad_x = np.pad(
            x,
            (
                (0, 0),
                (0, 0),
                (self.padding[0], self.padding[0]),
                (self.padding[1], self.padding[1]),
            ),
            "constant",
            constant_values=(self.min_val, self.min_val),
        )
        self.pad_x = pad_x
        self.pad_shape = pad_x.shape

        self.out_height = int((self.in_height - self.w_height) / self.stride[0]) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride[1]) + 1
        self.pad_out_height = np.uint16(
            math.ceil((self.pad_shape[2] - self.w_height + 1) / self.stride[0])
        )
        self.pad_out_width = np.uint16(
            math.ceil((self.pad_shape[3] - self.w_width + 1) / self.stride[1])
        )

        out = np.zeros(
            (self.in_batch, self.in_channel, self.pad_out_height, self.pad_out_width)
        )
        self.arg_max = np.zeros_like(out, dtype=np.int32)
        for n in range(self.in_batch):
            for c in range(self.in_channel):
                for i in range(self.pad_out_height):
                    for j in range(self.pad_out_width):
                        start_i = i * self.stride[0]
                        start_j = j * self.stride[1]
                        end_i = start_i + self.w_height
                        end_j = start_j + self.w_width
                        out[n, c, i, j] = np.max(
                            pad_x[n, c, start_i:end_i, start_j:end_j]
                        )
                        self.arg_max[n, c, i, j] = np.argmax(
                            pad_x[n, c, start_i:end_i, start_j:end_j]
                        )

        self.arg_max = self.arg_max
        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.pad_x)
        for n in range(self.in_batch):
            for c in range(self.in_channel):
                for i in range(self.pad_out_height):
                    for j in range(self.pad_out_width):
                        start_i = i * self.stride[0]
                        start_j = j * self.stride[1]
                        end_i = start_i + self.w_height
                        end_j = start_j + self.w_width
                        index = np.unravel_index(
                            self.arg_max[n, c, i, j], self.kernel_size
                        )
                        dx[n, c, start_i:end_i, start_j:end_j][index] = d_loss[
                            n, c, i, j
                        ]
        dx = dx[
            :,
            :,
            self.padding[0] : self.pad_shape[2] - self.padding[0],
            self.padding[1] : self.pad_shape[3] - self.padding[1],
        ]
        return dx


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestPoolingModule(flow.unittest.TestCase):
    def test_maxpool2d(test_case):
        input_arr = np.random.randn(6, 4, 7, 9)
        kernel_size, stride, padding = (4, 4), (1, 1), (1, 2)

        m_numpy = MaxPool2dNumpy(kernel_size, stride, padding)
        numpy_output = m_numpy(input_arr)

        m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        x = flow.Tensor(input_arr)
        output = m(x)
        test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))

    def test_maxpool2d_special_kernel_size(test_case):
        input_arr = np.random.randn(1, 1, 6, 6)
        kernel_size, stride, padding = (1, 1), (5, 5), (0, 0)

        m_numpy = MaxPool2dNumpy(kernel_size, stride, padding)
        numpy_output = m_numpy(input_arr)

        m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        x = flow.Tensor(input_arr)
        output = m(x)
        test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))

    def test_maxpool2d_diff_kernel_stride(test_case):
        input_arr = np.random.randn(9, 7, 32, 20)
        kernel_size, stride, padding = (2, 3), (4, 5), (1, 2)

        m_numpy = MaxPool2dNumpy(kernel_size, stride, padding)
        numpy_output = m_numpy(input_arr)

        m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        x = flow.Tensor(input_arr)
        output = m(x)

        test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))

    def test_maxpool2d_negative_input(test_case):
        input_arr = -1.23456 * np.ones((1, 1, 1, 1), dtype=np.float)
        kernel_size, stride, padding = (5, 5), (5, 5), (2, 2)

        m_numpy = MaxPool2dNumpy(kernel_size, stride, padding)
        numpy_output = m_numpy(input_arr)

        m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        x = flow.Tensor(input_arr)
        output = m(x)
        test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()
