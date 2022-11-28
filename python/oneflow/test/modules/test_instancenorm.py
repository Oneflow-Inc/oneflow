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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_instancenorm1d(test_case, device):
    input_arr = np.array(
        [
            [
                [-0.1091, 2.0041, 0.885, -0.0412],
                [-1.2055, 0.7442, 2.33, 1.2411],
                [-1.2466, 0.3667, 1.2267, 0.3043],
            ],
            [
                [-0.2484, -1.1407, 0.3352, 0.6687],
                [-0.2975, -0.0227, -0.2302, -0.3762],
                [-0.7759, -0.6789, 1.1444, 1.8077],
            ],
        ],
        dtype=np.float32,
    )
    output_arr = np.array(
        [
            [
                [-0.9262, 1.5395, 0.2337, -0.847],
                [-1.5486, -0.026, 1.2125, 0.3621],
                [-1.5807, 0.2287, 1.1933, 0.1587],
            ],
            [
                [-0.2215, -1.5212, 0.6285, 1.1143],
                [-0.5016, 1.5917, 0.011, -1.1011],
                [-1.0207, -0.9346, 0.6833, 1.2719],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.InstanceNorm1d(num_features=3, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.tensor(input_arr, dtype=flow.float32, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output_arr, rtol=1e-3, atol=1e-3))
    m.eval()
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output_arr, rtol=1e-3, atol=1e-3))


def _test_instancenorm2d(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [-0.8791, 0.2553, 0.7403, -0.2859],
                    [0.8006, -1.7701, -0.9617, 0.1705],
                    [0.2842, 1.7825, 0.3365, -0.8525],
                ],
                [
                    [0.7332, -0.0737, 0.7245, -0.6551],
                    [1.4461, -0.1827, 0.9737, -2.1571],
                    [0.4657, 0.7244, 0.3378, 0.1775],
                ],
            ],
            [
                [
                    [1.8896, 1.8686, 0.1896, 0.9817],
                    [-0.0671, 1.5569, 1.1449, 0.0086],
                    [-0.9468, -0.0124, 1.3227, -0.6567],
                ],
                [
                    [-0.8472, 1.3012, -1.1065, 0.9348],
                    [1.0346, 1.5703, 0.2419, -0.7048],
                    [0.6957, -0.4523, -0.8819, 1.0164],
                ],
            ],
        ],
        dtype=np.float32,
    )
    output = np.array(
        [
            [
                [
                    [-0.9155, 0.31, 0.8339, -0.2747],
                    [0.8991, -1.8781, -1.0048, 0.2183],
                    [0.3412, 1.9598, 0.3977, -0.8868],
                ],
                [
                    [0.586, -0.3169, 0.5763, -0.9675],
                    [1.3837, -0.4389, 0.8551, -2.6483],
                    [0.2867, 0.5761, 0.1435, -0.0358],
                ],
            ],
            [
                [
                    [1.374, 1.3515, -0.4466, 0.4017],
                    [-0.7215, 1.0177, 0.5765, -0.6405],
                    [-1.6636, -0.663, 0.7669, -1.353],
                ],
                [
                    [-1.1583, 1.1444, -1.4363, 0.7516],
                    [0.8586, 1.4328, 0.009, -1.0057],
                    [0.4954, -0.7351, -1.1955, 0.8391],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.InstanceNorm2d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.tensor(input_arr, dtype=flow.float32, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 0.0001, 0.0001))
    m.eval()
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 0.0001, 0.0001))


def _test_instancenorm3d(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [
                        [1.04569761, 0.22863248, 1.42439335, 1.62249689],
                        [-0.80578825, -0.27276461, 1.04556507, 0.56864134],
                        [-1.24085419, -1.23960097, 0.33451416, -1.84820402],
                    ],
                    [
                        [-1.511261, 1.06157517, -0.26715858, -1.32888141],
                        [1.17976881, -0.07931171, 0.33910684, -1.93458573],
                        [-1.72659647, 0.79049652, 0.39102785, -1.16264882],
                    ],
                ],
                [
                    [
                        [0.30067973, -1.2912226, -0.61508225, 0.56454001],
                        [0.87074187, -1.69257376, 0.36119148, -0.31014289],
                        [0.20776964, 1.26195488, -1.37122193, -0.17945234],
                    ],
                    [
                        [-0.31112407, -0.80682631, 0.8233194, 0.6384975],
                        [0.57617527, 0.45505028, 1.68286151, -1.09590744],
                        [-1.18127546, -1.07529277, 0.52779943, 1.21755926],
                    ],
                ],
            ],
            [
                [
                    [
                        [-0.12832351, 1.05625455, -0.23253249, -0.64747611],
                        [-0.00738123, -1.41390089, -1.92664144, -0.21427625],
                        [-0.94631219, -0.86493989, 0.21026905, 0.24989732],
                    ],
                    [
                        [1.3859182, 1.72002107, 0.50091892, 1.04198896],
                        [0.71694594, 1.66417023, -1.63030052, 0.77182641],
                        [0.71545083, 1.96458366, -1.99031931, 1.3196714],
                    ],
                ],
                [
                    [
                        [1.80091702, 0.02834973, 0.82259214, -1.05597501],
                        [-0.58212207, 0.44205949, -0.14740003, -0.994508],
                        [1.14678114, -0.39196097, 1.2554798, -0.41829324],
                    ],
                    [
                        [-1.0153903, -0.25755713, -1.81756333, -1.06781159],
                        [1.79680841, -1.9107133, -0.64325796, -1.94640775],
                        [1.30671156, 1.20445339, -1.26262901, -0.79494188],
                    ],
                ],
            ],
        ],
        dtype=np.float32,
    )
    output_arr = np.array(
        [
            [
                [
                    [
                        [1.067, 0.3324, 1.4075, 1.5856],
                        [-0.5976, -0.1184, 1.0669, 0.6381],
                        [-0.9888, -0.9877, 0.4276, -1.5349],
                    ],
                    [
                        [-1.2319, 1.0813, -0.1134, -1.068],
                        [1.1876, 0.0555, 0.4317, -1.6126],
                        [-1.4256, 0.8376, 0.4784, -0.9185],
                    ],
                ],
                [
                    [
                        [0.3447, -1.3751, -0.6446, 0.6298],
                        [0.9606, -1.8087, 0.4101, -0.3152],
                        [0.2444, 1.3833, -1.4615, -0.174],
                    ],
                    [
                        [-0.3162, -0.8518, 0.9094, 0.7097],
                        [0.6424, 0.5115, 1.838, -1.1641],
                        [-1.2563, -1.1418, 0.5901, 1.3353],
                    ],
                ],
            ],
            [
                [
                    [
                        [-0.2327, 0.8016, -0.3236, -0.6859],
                        [-0.1271, -1.3551, -1.8028, -0.3077],
                        [-0.9469, -0.8758, 0.063, 0.0976],
                    ],
                    [
                        [1.0895, 1.3812, 0.3167, 0.7892],
                        [0.5054, 1.3324, -1.5441, 0.5533],
                        [0.5041, 1.5947, -1.8584, 1.0316],
                    ],
                ],
                [
                    [
                        [1.7507, 0.1901, 0.8894, -0.7645],
                        [-0.3473, 0.5544, 0.0354, -0.7104],
                        [1.1748, -0.1799, 1.2705, -0.2031],
                    ],
                    [
                        [-0.7288, -0.0616, -1.435, -0.7749],
                        [1.7471, -1.517, -0.4012, -1.5485],
                        [1.3156, 1.2256, -0.9465, -0.5347],
                    ],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.InstanceNorm3d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.tensor(input_arr, dtype=flow.float32, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output_arr, 0.0001, 0.0001))
    m.eval()
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output_arr, 0.0001, 0.0001))


def _test_instancenorm1d_backward(test_case, device):
    input_arr = np.array(
        [
            [
                [-0.1091, 2.0041, 0.885, -0.0412],
                [-1.2055, 0.7442, 2.33, 1.2411],
                [-1.2466, 0.3667, 1.2267, 0.3043],
            ],
            [
                [-0.2484, -1.1407, 0.3352, 0.6687],
                [-0.2975, -0.0227, -0.2302, -0.3762],
                [-0.7759, -0.6789, 1.1444, 1.8077],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.InstanceNorm1d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.tensor(input_arr, device=flow.device(device), requires_grad=True)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.zeros(shape=input_arr.shape), 1e-05, 1e-05)
    )


def _test_instancenorm2d_backward(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [-0.8791, 0.2553, 0.7403, -0.2859],
                    [0.8006, -1.7701, -0.9617, 0.1705],
                    [0.2842, 1.7825, 0.3365, -0.8525],
                ],
                [
                    [0.7332, -0.0737, 0.7245, -0.6551],
                    [1.4461, -0.1827, 0.9737, -2.1571],
                    [0.4657, 0.7244, 0.3378, 0.1775],
                ],
            ],
            [
                [
                    [1.8896, 1.8686, 0.1896, 0.9817],
                    [-0.0671, 1.5569, 1.1449, 0.0086],
                    [-0.9468, -0.0124, 1.3227, -0.6567],
                ],
                [
                    [-0.8472, 1.3012, -1.1065, 0.9348],
                    [1.0346, 1.5703, 0.2419, -0.7048],
                    [0.6957, -0.4523, -0.8819, 1.0164],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.InstanceNorm2d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.tensor(input_arr, device=flow.device(device), requires_grad=True)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.zeros(shape=input_arr.shape), 1e-05, 1e-05)
    )


def _test_instancenorm3d_backward(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [
                        [1.04569761, 0.22863248, 1.42439335, 1.62249689],
                        [-0.80578825, -0.27276461, 1.04556507, 0.56864134],
                        [-1.24085419, -1.23960097, 0.33451416, -1.84820402],
                    ],
                    [
                        [-1.511261, 1.06157517, -0.26715858, -1.32888141],
                        [1.17976881, -0.07931171, 0.33910684, -1.93458573],
                        [-1.72659647, 0.79049652, 0.39102785, -1.16264882],
                    ],
                ],
                [
                    [
                        [0.30067973, -1.2912226, -0.61508225, 0.56454001],
                        [0.87074187, -1.69257376, 0.36119148, -0.31014289],
                        [0.20776964, 1.26195488, -1.37122193, -0.17945234],
                    ],
                    [
                        [-0.31112407, -0.80682631, 0.8233194, 0.6384975],
                        [0.57617527, 0.45505028, 1.68286151, -1.09590744],
                        [-1.18127546, -1.07529277, 0.52779943, 1.21755926],
                    ],
                ],
            ],
            [
                [
                    [
                        [-0.12832351, 1.05625455, -0.23253249, -0.64747611],
                        [-0.00738123, -1.41390089, -1.92664144, -0.21427625],
                        [-0.94631219, -0.86493989, 0.21026905, 0.24989732],
                    ],
                    [
                        [1.3859182, 1.72002107, 0.50091892, 1.04198896],
                        [0.71694594, 1.66417023, -1.63030052, 0.77182641],
                        [0.71545083, 1.96458366, -1.99031931, 1.3196714],
                    ],
                ],
                [
                    [
                        [1.80091702, 0.02834973, 0.82259214, -1.05597501],
                        [-0.58212207, 0.44205949, -0.14740003, -0.994508],
                        [1.14678114, -0.39196097, 1.2554798, -0.41829324],
                    ],
                    [
                        [-1.0153903, -0.25755713, -1.81756333, -1.06781159],
                        [1.79680841, -1.9107133, -0.64325796, -1.94640775],
                        [1.30671156, 1.20445339, -1.26262901, -0.79494188],
                    ],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.InstanceNorm3d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.tensor(input_arr, device=flow.device(device), requires_grad=True)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.zeros(shape=input_arr.shape), 1e-05, 1e-05)
    )


@flow.unittest.skip_unless_1n1d()
class TestInstanceNorm(flow.unittest.TestCase):
    def test_instancenorm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_instancenorm1d,
            _test_instancenorm2d,
            _test_instancenorm3d,
            _test_instancenorm1d_backward,
            _test_instancenorm2d_backward,
            _test_instancenorm3d_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    # NOTE: in the following tese cases, if set track_running_stats=True, will fail!
    # it could be some bud to be fixed in nn.InstanceNorm
    @autotest(n=5, auto_backward=True, rtol=1e-3, atol=1e-3, check_graph=True)
    def test_instancenorm_with_random_data(test_case):
        height = random(1, 6).to(int)
        width = random(1, 6).to(int)
        m = torch.nn.InstanceNorm1d(
            num_features=height,
            eps=random().to(float) | nothing(),
            momentum=random().to(float) | nothing(),
            affine=random().to(bool),
            track_running_stats=False,
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=3, dim1=height, dim2=width).to(device)
        y = m(x)
        return y

    @autotest(n=5, rtol=1e-3, atol=1e-3)
    def test_instancenorm_with_random_data2(test_case):
        channel = random(1, 6).to(int)
        height = random(1, 6).to(int)
        width = random(1, 6).to(int)
        m = torch.nn.InstanceNorm2d(
            num_features=channel,
            eps=random().to(float) | nothing(),
            momentum=random().to(float) | nothing(),
            affine=random().to(bool),
            track_running_stats=False,
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=4, dim1=channel, dim2=height, dim3=width).to(device)
        y = m(x)
        return y

    @autotest(n=5, rtol=1e-3, atol=1e-3)
    def test_instancenorm_with_random_data3(test_case):
        channel = random(1, 6).to(int)
        depth = random(1, 6).to(int)
        height = random(1, 6).to(int)
        width = random(1, 6).to(int)
        m = torch.nn.InstanceNorm3d(
            num_features=channel,
            eps=random().to(float) | nothing(),
            momentum=random().to(float) | nothing(),
            affine=random().to(bool),
            track_running_stats=False,
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=5, dim1=channel, dim2=depth, dim3=height, dim4=width).to(
            device
        )
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()
