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


g_samples = [
    {
        "kernel": (3, 2),
        "padding": 0,
        "stride": (2, 1),
        "in": np.array(
            [
                [
                    [
                        [
                            -0.1953,
                            1.3992,
                            -0.7464,
                            0.6910,
                            -1.5484,
                            0.4970,
                            1.4963,
                            0.3080,
                            -1.4730,
                            -0.1238,
                        ],
                        [
                            -0.3532,
                            1.2078,
                            -0.3796,
                            0.7326,
                            -1.5795,
                            0.2128,
                            0.6501,
                            -0.1266,
                            -1.3121,
                            0.1483,
                        ],
                        [
                            -0.3412,
                            -1.6446,
                            -1.0039,
                            -0.5594,
                            0.7450,
                            -0.5323,
                            -1.6887,
                            0.2399,
                            1.9422,
                            0.4214,
                        ],
                        [
                            -1.6362,
                            -1.2234,
                            -1.2531,
                            0.6109,
                            0.2228,
                            -0.2080,
                            0.6359,
                            0.2451,
                            0.3864,
                            0.4263,
                        ],
                        [
                            0.7053,
                            0.3413,
                            0.9090,
                            -0.4057,
                            -0.2830,
                            1.0444,
                            -0.2884,
                            0.7638,
                            -1.4793,
                            0.2079,
                        ],
                        [
                            -0.1207,
                            0.8458,
                            -0.9521,
                            0.3630,
                            0.1772,
                            0.3945,
                            0.4056,
                            -0.7822,
                            0.6166,
                            1.3343,
                        ],
                        [
                            -0.4115,
                            0.5802,
                            1.2909,
                            1.6508,
                            -0.0561,
                            -0.7964,
                            0.9786,
                            0.4265,
                            0.7262,
                            0.2819,
                        ],
                        [
                            -0.2667,
                            -0.0792,
                            0.4771,
                            0.3248,
                            -0.1313,
                            -0.3325,
                            -0.9973,
                            0.3128,
                            -0.5151,
                            -0.1225,
                        ],
                        [
                            -1.4983,
                            0.2604,
                            -0.9127,
                            0.0822,
                            0.3708,
                            -2.6024,
                            0.2249,
                            -0.7500,
                            0.3152,
                            0.1931,
                        ],
                        [
                            -0.2171,
                            -0.2602,
                            0.9051,
                            -0.0933,
                            -0.0902,
                            -1.3837,
                            -1.2519,
                            -1.3091,
                            0.7155,
                            2.3376,
                        ],
                    ]
                ]
            ]
        ),
        "out": np.array(
            [
                [
                    [
                        [
                            0.0121,
                            -0.1946,
                            -0.2110,
                            -0.2531,
                            -0.3675,
                            0.1059,
                            0.1465,
                            -0.0703,
                            -0.0662,
                        ],
                        [
                            -0.6331,
                            -0.6458,
                            -0.2837,
                            0.0551,
                            0.1648,
                            -0.1729,
                            -0.0154,
                            0.3497,
                            0.3175,
                        ],
                        [
                            0.3234,
                            0.5025,
                            0.4760,
                            0.2410,
                            0.0801,
                            0.2897,
                            0.2506,
                            0.0453,
                            0.2813,
                        ],
                        [
                            -0.2359,
                            0.2694,
                            0.4855,
                            0.3735,
                            -0.5913,
                            -0.5875,
                            0.0326,
                            0.0859,
                            0.1465,
                        ],
                    ]
                ]
            ]
        ),
    }
]


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_AvgPool2d(test_case):
        global g_samples
        for sample in g_samples:
            of_avgpool2d = flow.nn.AvgPool2d(
                kernel_size=sample["kernel"],
                padding=sample["padding"],
                stride=sample["stride"],
            )
            x = flow.Tensor(sample["in"])
            of_y = of_avgpool2d(x)
            test_case.assertTrue(of_y.numpy().shape == sample["out"].shape)
            test_case.assertTrue(np.allclose(of_y.numpy(), sample["out"], 1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()
