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
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


def _test_groupnorm(test_case, device):
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
                    [-1.0548115, 0.18125379, 0.7097197, -0.4084487],
                    [0.77542377, -2.0256634, -1.1448141, 0.08885399],
                    [0.21274385, 1.845322, 0.26973096, -1.0258276],
                ],
                [
                    [0.7019834, -0.17723128, 0.6925037, -0.81073654],
                    [1.4787737, -0.2959999, 0.96403706, -2.4473464],
                    [0.4105099, 0.69239473, 0.2711475, 0.09648134],
                ],
            ],
            [
                [
                    [1.5438884, 1.5218256, -0.24213786, 0.5900453],
                    [-0.5118278, 1.1943525, 0.76150376, -0.43229714],
                    [-1.4360437, -0.4543598, 0.94830114, -1.1312639],
                ],
                [
                    [-1.3314037, 0.9257132, -1.6038253, 0.54077196],
                    [0.6456222, 1.2084305, -0.18719131, -1.1817979],
                    [0.28957263, -0.91652036, -1.3678597, 0.6265012],
                ],
            ],
        ],
        dtype=np.float32,
    )
    x = flow.tensor(input_arr, dtype=flow.float32, device=flow.device(device))
    m = flow.nn.GroupNorm(num_groups=1, num_channels=2).to(device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-03, 1e-03))


def _test_groupnorm_3d(test_case, device):
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
    output = np.array(
        [
            [
                [
                    [
                        [1.0670303, 0.3324034, 1.4075173, 1.5856332],
                        [-0.5976489, -0.11840499, 1.0669112, 0.6381069],
                        [-0.9888186, -0.9876919, 0.42760208, -1.5348896],
                    ],
                    [
                        [-1.2319425, 1.0813059, -0.11336456, -1.0679643],
                        [1.1875744, 0.05552938, 0.43173137, -1.6125557],
                        [-1.4255517, 0.8375778, 0.4784138, -0.9185038],
                    ],
                ],
                [
                    [
                        [0.3447361, -1.3750811, -0.6446106, 0.62979853],
                        [0.9606047, -1.8086823, 0.41011015, -0.3151683],
                        [0.24436034, 1.3832531, -1.4615086, -0.17397629],
                    ],
                    [
                        [-0.31622827, -0.8517619, 0.9093717, 0.7096987],
                        [0.6423687, 0.51151085, 1.8379811, -1.1640717],
                        [-1.2562994, -1.1418006, 0.59010565, 1.3352901],
                    ],
                ],
            ],
            [
                [
                    [
                        [-0.23265934, 0.8016156, -0.32364592, -0.6859402],
                        [-0.12706259, -1.3551185, -1.802801, -0.30770612],
                        [-0.946859, -0.8758114, 0.06297152, 0.09757163],
                    ],
                    [
                        [1.0894505, 1.3811613, 0.3167428, 0.78916013],
                        [0.50535965, 1.3323971, -1.5440607, 0.55327666],
                        [0.50405425, 1.5946931, -1.8583992, 1.0316093],
                    ],
                ],
                [
                    [
                        [1.7506906, 0.19012147, 0.8893728, -0.7645185],
                        [-0.3473382, 0.5543517, 0.03539129, -0.71040297],
                        [1.174789, -0.17992027, 1.2704874, -0.20310321],
                    ],
                    [
                        [-0.7287877, -0.06159106, -1.4350212, -0.7749395],
                        [1.7470733, -1.5170306, -0.40116227, -1.548456],
                        [1.3155918, 1.2255636, -0.9464568, -0.53470486],
                    ],
                ],
            ],
        ],
        dtype=np.float32,
    )
    x = flow.tensor(input_arr, dtype=flow.float32, device=flow.device(device))
    m = flow.nn.GroupNorm(num_groups=2, num_channels=2, affine=False).to(
        device=flow.device(device)
    )
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-03, 1e-03))


def _test_groupnorm_backward(test_case, device):
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
    x = flow.tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m = flow.nn.GroupNorm(num_groups=1, num_channels=2).to(device=flow.device(device))
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.zeros(shape=input_arr.shape), 1e-03, 1e-03)
    )


def _test_groupnorm_backward_fp16(test_case, device):
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
        dtype=np.float16,
    )
    x = flow.tensor(
        input_arr, dtype=flow.float16, device=flow.device(device), requires_grad=True
    )
    m = (
        flow.nn.GroupNorm(num_groups=1, num_channels=2)
        .to(device=flow.device(device))
        .to(flow.float16)
    )
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.zeros(shape=input_arr.shape), 1e-03, 1e-03)
    )


def _test_groupnorm_backward_3d(test_case, device):
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
    x = flow.tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m = flow.nn.GroupNorm(num_groups=2, num_channels=2, affine=False).to(
        device=flow.device(device)
    )
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.zeros(shape=input_arr.shape), 1e-03, 1e-03)
    )


def _test_groupnorm_backward_3d_fp16(test_case, device):
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
        dtype=np.float16,
    )
    x = flow.tensor(
        input_arr, dtype=flow.float16, device=flow.device(device), requires_grad=True
    )
    m = (
        flow.nn.GroupNorm(num_groups=2, num_channels=2, affine=False)
        .to(device=flow.device(device))
        .to(flow.float16)
    )
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.zeros(shape=input_arr.shape), 1e-03, 1e-03)
    )


def _test_groupnorm_nhwc(test_case, shape, num_groups):
    (n, c, h, w) = shape
    x = flow.tensor(
        np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)
    ).to("cuda")
    gamma = flow.tensor(
        np.random.uniform(low=0.0, high=1.0, size=(c)).astype(np.float32)
    ).to("cuda")
    beta = flow.tensor(
        np.random.uniform(low=0.0, high=1.0, size=(c)).astype(np.float32)
    ).to("cuda")
    y = flow._C.group_norm(x, gamma, beta, True, num_groups, 1e-5)
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    y_nhwc = flow._C.group_norm(
        x_nhwc, gamma, beta, True, num_groups, 1e-5, "channels_last"
    )
    test_case.assertTrue(
        np.allclose(y_nhwc.permute(0, 3, 1, 2).numpy(), y, 1e-03, 1e-03)
    )


@flow.unittest.skip_unless_1n1d()
class TestGroupNorm(flow.unittest.TestCase):
    def test_groupnorm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_groupnorm,
            _test_groupnorm_3d,
            _test_groupnorm_backward,
            _test_groupnorm_backward_3d,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_groupnorm_grad_fp16(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_groupnorm_backward_fp16,
            _test_groupnorm_backward_3d_fp16,
        ]
        # cpu test will raise error: var only support floating point dtypes
        # https://github.com/Oneflow-Inc/oneflow/issues/9559
        # arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(rtol=1e-03, atol=1e-03, check_graph=True)
    def test_group_norm_with_random_data(test_case):
        channels = random(5, 20)
        m = torch.nn.GroupNorm(
            num_groups=random(1, 5),
            num_channels=channels,
            eps=random(0, 1) | nothing(),
            affine=random(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=4, dim1=channels).to(device)
        y = m(x)
        return y

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_groupnorm_nhwc(test_case):
        _test_groupnorm_nhwc(test_case, (16, 64, 128, 128), 32)


if __name__ == "__main__":
    unittest.main()
