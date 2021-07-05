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
from collections import OrderedDict

import numpy as np
import math

import oneflow.experimental as flow
from test_util import GenArgList


def _nd_tuple_to_dhw(nd_tuple, dim, prefix=1, dhw_offset=0):
    assert dim <= 3
    assert dim == len(nd_tuple) - dhw_offset
    nd_tuple = list(nd_tuple)
    dhw_tuple = nd_tuple[:dhw_offset]
    dhw_tuple.extend([prefix for _ in range(3 - dim)])
    dhw_tuple.extend(nd_tuple[dhw_offset:])
    return tuple(dhw_tuple)


def _dhw_tuple_to_nd(dhw_tuple, dim, prefix=1, dhw_offset=0):
    assert dim <= 3
    assert 3 == len(dhw_tuple) - dhw_offset
    dhw_tuple = list(dhw_tuple)
    nd_tuple = dhw_tuple[:dhw_offset]
    nd_offset = dhw_offset + 3 - dim
    for i in dhw_tuple[dhw_offset:nd_offset]:
        assert prefix == i
    nd_tuple.extend(dhw_tuple[nd_offset:])
    return tuple(nd_tuple)


class AvgPoolNumpy:
    def __init__(self, dim=2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)):
        self.dim = dim
        self.stride = _nd_tuple_to_dhw(stride, dim)
        self.padding = _nd_tuple_to_dhw(padding, dim, prefix=0)
        self.kernel_size = _nd_tuple_to_dhw(kernel_size, dim)
        self.w_depth = self.kernel_size[0]
        self.w_height = self.kernel_size[1]
        self.w_width = self.kernel_size[2]
        self.min_val = 0.0

    def __call__(self, x):
        self.x_shape = x.shape
        x_shape_5d = _nd_tuple_to_dhw(self.x_shape, self.dim, prefix=1, dhw_offset=2)
        x = x.reshape(x_shape_5d)
        self.in_batch = np.shape(x)[0]
        self.in_channel = np.shape(x)[1]
        self.in_depth = np.shape(x)[2]
        self.in_height = np.shape(x)[3]
        self.in_width = np.shape(x)[4]

        pad_x = np.pad(
            x,
            (
                (0, 0),
                (0, 0),
                (self.padding[0], self.padding[0]),
                (self.padding[1], self.padding[1]),
                (self.padding[2], self.padding[2]),
            ),
            "constant",
            constant_values=(self.min_val, self.min_val),
        )
        self.pad_x = pad_x
        self.pad_shape = pad_x.shape

        self.out_depth = int((self.in_depth - self.w_depth) / self.stride[0]) + 1
        self.out_height = int((self.in_height - self.w_height) / self.stride[1]) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride[2]) + 1
        self.pad_out_depth = np.uint16(
            math.ceil((self.pad_shape[2] - self.w_depth + 1) / self.stride[0])
        )
        self.pad_out_height = np.uint16(
            math.ceil((self.pad_shape[3] - self.w_height + 1) / self.stride[1])
        )
        self.pad_out_width = np.uint16(
            math.ceil((self.pad_shape[4] - self.w_width + 1) / self.stride[2])
        )

        out = np.zeros(
            (
                self.in_batch,
                self.in_channel,
                self.pad_out_depth,
                self.pad_out_height,
                self.pad_out_width,
            )
        )
        self.arg_avg = np.zeros_like(out)
        for n in range(self.in_batch):
            for c in range(self.in_channel):
                for i in range(self.pad_out_depth):
                    for j in range(self.pad_out_height):
                        for k in range(self.pad_out_width):
                            start_i = i * self.stride[0]
                            start_j = j * self.stride[1]
                            start_k = k * self.stride[2]
                            end_i = start_i + self.w_depth
                            end_j = start_j + self.w_height
                            end_k = start_k + self.w_width
                            out[n, c, i, j, k] = np.average(
                                pad_x[n, c, start_i:end_i, start_j:end_j, start_k:end_k]
                            )
                            self.arg_avg[n, c, i, j, k] = np.average(
                                pad_x[n, c, start_i:end_i, start_j:end_j, start_k:end_k]
                            )

        self.out_shape_5d = out.shape
        out_shape = _dhw_tuple_to_nd(out.shape, self.dim, dhw_offset=2)
        out = out.reshape(out_shape)
        return out


def _test_avgpool3d(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [[-1.1132425, -0.79719835], [1.99409501, 0.23270504]],
                    [[-0.69827855, -0.19336448], [0.86132664, -0.86734113]],
                ],
                [
                    [[0.90614991, -1.11548232], [-0.17957948, -0.14095705]],
                    [[0.12856562, -0.82078871], [-0.79095713, -0.86583306]],
                ],
            ],
            [
                [
                    [[-1.99924145, 0.39951706], [-1.31197624, -0.68801404]],
                    [[-0.09358264, 0.12486073], [-0.45929356, 0.31948792]],
                ],
                [
                    [[0.72989192, 1.65362442], [0.12919752, -1.45644394]],
                    [[-0.33608345, -0.4950027], [-0.30841882, 1.06204887]],
                ],
            ],
        ]
    )
    dim = 3
    kernel_size, stride, padding = (2, 2, 2), (1, 1, 1), (0, 0, 0)
    m_numpy = AvgPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))


def _test_avgpool3d_backward(test_case, device):
    dim = 3
    input_arr = np.array(
        [
            [
                [
                    [[-1.1132425, -0.79719835], [1.99409501, 0.23270504]],
                    [[-0.69827855, -0.19336448], [0.86132664, -0.86734113]],
                ],
                [
                    [[0.90614991, -1.11548232], [-0.17957948, -0.14095705]],
                    [[0.12856562, -0.82078871], [-0.79095713, -0.86583306]],
                ],
            ],
            [
                [
                    [[-1.99924145, 0.39951706], [-1.31197624, -0.68801404]],
                    [[-0.09358264, 0.12486073], [-0.45929356, 0.31948792]],
                ],
                [
                    [[0.72989192, 1.65362442], [0.12919752, -1.45644394]],
                    [[-0.33608345, -0.4950027], [-0.30841882, 1.06204887]],
                ],
            ],
        ]
    )
    kernel_size, stride, padding = (2, 2, 2), (1, 1, 1), (0, 0, 0)
    m_numpy = AvgPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = np.zeros(shape=input_arr.shape)
    numpy_grad[...] = 0.125
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-5, 1e-5))


def _test_avgpool3d_special_kernel_size_backward(test_case, device):
    dim = 3
    input_arr = np.array(
        [
            [
                [
                    [
                        [
                            1.66918755,
                            -0.91884044,
                            -0.53434356,
                            -0.57682845,
                            -0.57808441,
                            1.99174729,
                        ],
                        [
                            -0.57801338,
                            1.810334,
                            -0.30454292,
                            -0.32011417,
                            -2.4486984,
                            -0.66338876,
                        ],
                        [
                            -0.15772485,
                            0.6784365,
                            1.18897709,
                            1.20692234,
                            1.43578745,
                            -0.36833255,
                        ],
                        [
                            0.74718159,
                            0.09179258,
                            -0.94193085,
                            -0.35707129,
                            -0.62257021,
                            0.42824892,
                        ],
                        [
                            -0.13482852,
                            -0.02991985,
                            0.28971932,
                            1.80695194,
                            -0.07023364,
                            -0.92182529,
                        ],
                        [
                            -0.02296651,
                            -1.43817104,
                            1.4028344,
                            0.18194114,
                            -0.59439764,
                            1.51888284,
                        ],
                    ],
                    [
                        [
                            0.39941812,
                            -0.69972636,
                            1.05458831,
                            0.93664904,
                            -1.00730994,
                            1.09524098,
                        ],
                        [
                            0.63022077,
                            0.85397415,
                            1.0084123,
                            -0.20605707,
                            -0.37284122,
                            0.11387859,
                        ],
                        [
                            -1.26611431,
                            -0.62012754,
                            0.09563748,
                            -0.21232549,
                            -1.77755391,
                            0.22544966,
                        ],
                        [
                            0.05055287,
                            -0.97104387,
                            0.00743758,
                            -0.01799878,
                            -0.01687093,
                            -0.95385641,
                        ],
                        [
                            -0.46048377,
                            0.74474033,
                            0.38518884,
                            1.4415209,
                            -0.74031676,
                            1.3467917,
                        ],
                        [
                            1.07532674,
                            -1.22199077,
                            0.53129623,
                            -1.15805626,
                            -1.59087007,
                            0.27252823,
                        ],
                    ],
                    [
                        [
                            2.10041429,
                            -2.43180683,
                            1.21660805,
                            -2.60185516,
                            1.05938698,
                            0.96355525,
                        ],
                        [
                            -1.25661354,
                            -1.13195752,
                            0.47894153,
                            1.19304616,
                            -0.69451204,
                            2.2175799,
                        ],
                        [
                            1.34278748,
                            -1.52081064,
                            -0.2507571,
                            0.67087564,
                            -0.79763021,
                            -0.41767333,
                        ],
                        [
                            -2.32956058,
                            0.03233625,
                            -1.47391582,
                            0.70333218,
                            -0.2506578,
                            0.24757612,
                        ],
                        [
                            0.22672213,
                            -0.60840215,
                            -1.55909351,
                            -0.30993582,
                            -0.25493395,
                            -1.13345972,
                        ],
                        [
                            -0.30647421,
                            -0.48087784,
                            -0.71393674,
                            -1.36828179,
                            1.10667612,
                            -0.15967295,
                        ],
                    ],
                    [
                        [
                            0.32983435,
                            -0.91425562,
                            -0.35299711,
                            1.31247588,
                            0.15367215,
                            -1.98610838,
                        ],
                        [
                            0.81303132,
                            0.15115689,
                            1.8122944,
                            0.96024569,
                            1.75029563,
                            1.79526488,
                        ],
                        [
                            -0.72335846,
                            -0.25343156,
                            0.68296792,
                            0.12407177,
                            0.2543815,
                            -0.51771794,
                        ],
                        [
                            -1.56714417,
                            1.19790861,
                            1.20180306,
                            0.41645108,
                            -0.4753875,
                            0.43112448,
                        ],
                        [
                            -0.72958873,
                            1.07136698,
                            0.99048707,
                            -1.65848592,
                            0.53776319,
                            0.37002138,
                        ],
                        [
                            1.45602655,
                            0.05036957,
                            0.53813642,
                            -1.29038552,
                            0.66232652,
                            -0.00563294,
                        ],
                    ],
                    [
                        [
                            1.82491436,
                            -1.87574983,
                            -0.27483037,
                            -1.41977775,
                            0.95369067,
                            -0.19138531,
                        ],
                        [
                            -1.25252398,
                            1.33494634,
                            -0.13758054,
                            -0.33883371,
                            1.80729216,
                            1.29806594,
                        ],
                        [
                            0.77033134,
                            -1.30258535,
                            -1.8302794,
                            0.52123884,
                            0.90620194,
                            -0.67787233,
                        ],
                        [
                            -0.29091427,
                            -0.27677645,
                            -0.18344966,
                            -0.92565511,
                            0.19842833,
                            0.59580347,
                        ],
                        [
                            -0.29520923,
                            0.17046046,
                            -0.80503485,
                            0.89908856,
                            0.69774822,
                            0.29579325,
                        ],
                        [
                            0.17788624,
                            -0.34228185,
                            -0.37028163,
                            -1.18220291,
                            1.77898418,
                            -0.17662215,
                        ],
                    ],
                    [
                        [
                            0.06161488,
                            1.56969206,
                            0.81895252,
                            -0.82887789,
                            0.9260089,
                            -0.0988148,
                        ],
                        [
                            0.21460429,
                            -1.4755581,
                            1.36994785,
                            1.17893958,
                            -1.01790093,
                            0.08058205,
                        ],
                        [
                            -0.78913355,
                            -0.48296865,
                            -1.08832194,
                            -0.81984527,
                            0.22901453,
                            0.0114611,
                        ],
                        [
                            -0.50999815,
                            -0.52438008,
                            -0.39893658,
                            -0.68719077,
                            1.0338822,
                            0.14097484,
                        ],
                        [
                            1.45503734,
                            1.70649681,
                            -0.53885203,
                            -0.62992688,
                            -0.3641152,
                            -0.1234822,
                        ],
                        [
                            -1.18950772,
                            1.64488172,
                            0.46651043,
                            -2.17475965,
                            0.36525702,
                            0.9185165,
                        ],
                    ],
                ]
            ]
        ]
    )
    kernel_size, stride, padding = (1, 1, 1), (5, 5, 5), (0, 0, 0)

    m_numpy = AvgPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = np.array(
        [
            [
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestPoolingModule(flow.unittest.TestCase):
    def test_avgpool3d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_avgpool3d,
            _test_avgpool3d_backward,
            _test_avgpool3d_special_kernel_size_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
