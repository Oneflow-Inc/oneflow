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

import oneflow.experimental as flow
import oneflow.experimental.nn as nn
from test_util import GenArgList


def _test_conv1d_bias_true(test_case, device):
    np_arr = np.array(
        [
            [
                [0.90499806, -1.11683071, 0.71605605, -0.56754625, 0.61944169],
                [-0.31317389, -0.26271924, 0.95579433, 0.52468461, 1.48926127],
            ]
        ]
    )
    input = flow.Tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    weight = np.array(
        [
            [
                [0.01997352, 0.23834395, 0.00526353],
                [-0.04861857, -0.22751901, -0.06725175],
            ],
            [
                [0.13344523, -0.35202524, 0.15168799],
                [-0.25714493, -0.17459838, 0.28768948],
            ],
            [
                [0.10671382, -0.28205597, -0.39752254],
                [0.36393702, 0.07843742, -0.33898622],
            ],
            [
                [0.20485674, 0.04222689, -0.18986180],
                [0.22519711, -0.15910202, -0.35057363],
            ],
        ]
    )
    bias = np.array([0.01012857, 0.38912651, -0.01600273, -0.38833040])
    m = nn.Conv1d(2, 4, 3, stride=1, bias=True)
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m.bias = flow.nn.Parameter(flow.Tensor(bias))
    m = m.to(device)
    np_out = np.array(
        [
            [
                [-0.22349545, -0.08447243, -0.37358052],
                [1.41303730, -0.04644597, 0.86949122],
                [-0.34765026, -0.31004351, -0.14158708],
                [-0.74985039, -0.87430149, -0.77354753],
            ]
        ]
    )
    output = m(input)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
    output = output.sum()
    output.backward()
    np_grad = np.array(
        [
            [
                [0.46498930, 0.11147892, -0.31895390, -0.78394318, -0.43043283],
                [0.28337064, -0.19941133, -0.66853344, -0.95190406, -0.46912211],
            ]
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


def _test_conv1d_group_bias_true(test_case, device):
    np_arr = np.array(
        [
            [
                [1.48566079, 0.54937589, 0.62353903, -0.94114172, -0.60260266],
                [0.61150503, -0.50289607, 1.41735041, -1.85877609, -1.04875529],
            ]
        ]
    )
    input = flow.Tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    weight = np.array(
        [
            [[0.25576305, 0.40814576, -0.05900212]],
            [[-0.24829513, 0.42756805, -0.01354307]],
            [[0.44658303, 0.46889144, 0.41060263]],
            [[0.30083328, -0.52216130, 0.12215579]],
        ]
    )
    bias = np.array([-0.03368823, -0.42125040, -0.42130581, -0.17434336])
    m = nn.Conv1d(2, 4, 3, groups=2, stride=1, bias=True)
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m.bias = flow.nn.Parameter(flow.Tensor(bias))
    m = m.to(device)
    np_out = np.array(
        [
            [
                [0.53372419, 0.41684598, -0.22277816],
                [-0.56368178, -0.27830642, -0.97031319],
                [0.19794616, -0.74452549, -1.09052706],
                [0.44534814, -1.29277706, 1.09451222],
            ]
        ]
    )
    output = m(input)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
    output = output.sum()
    output.backward()
    np_grad = np.array(
        [
            [
                [0.00746793, 0.84318173, 0.77063656, 0.76316863, -0.07254519],
                [0.74741632, 0.69414645, 1.22690487, 0.47948855, 0.53275841],
            ]
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


def _test_conv1d_group_large_out_bias_true(test_case, device):
    np_arr = np.array(
        [
            [
                [2.17964911, 0.91623521, 1.24746692, 0.73605931, -0.23738743],
                [-0.70412433, 0.10727754, 1.02078640, -0.09711888, -1.10814202],
            ]
        ]
    )
    input = flow.Tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    weight = np.array(
        [
            [[-2.07307473e-01, 1.28563240e-01, 3.71991515e-01]],
            [[-4.16422307e-01, 3.26921181e-05, -3.85845661e-01]],
            [[-1.82592362e-01, 1.43281639e-01, 4.19321984e-01]],
            [[-2.71174580e-01, 4.21470925e-02, 3.77335936e-01]],
            [[5.46190619e-01, -2.11819887e-01, -2.97858030e-01]],
            [[3.34832489e-01, 2.55918801e-01, -5.56600206e-02]],
        ]
    )
    bias = np.array(
        [-0.56865668, 0.17631066, -0.43992457, -0.24307285, -0.53672957, -0.52927947]
    )
    m = nn.Conv1d(2, 6, 3, groups=2, stride=1, bias=True)
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m.bias = flow.nn.Parameter(flow.Tensor(bias))
    m = m.to(device)
    np_out = np.array(
        [
            [
                [-0.43867296, -0.32441288, -0.82094181],
                [-1.21264362, -0.48919463, -0.25154343],
                [-0.18354186, -0.11983716, -0.66178048],
                [0.33756858, -0.26578707, -0.94211930],
                [-1.24808860, -0.66543078, 0.37145507],
                [-0.79440582, -0.22671542, -0.15066233],
            ]
        ]
    )
    output = m(input)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
    output = output.sum()
    output.backward()
    np_grad = np.array(
        [
            [
                [-0.80632210, -0.53444451, -0.12897667, 0.67734540, 0.40546784],
                [0.60984850, 0.69609451, 0.71991241, 0.11006390, 0.02381789],
            ]
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


def _test_conv1d_group_large_in_bias_true(test_case, device):
    np_arr = np.array(
        [
            [
                [0.73829210, 0.32275710, -0.73204273, -0.01697334, 1.72585976],
                [0.52866709, 0.28417364, 1.12931311, 1.73048413, -0.60748184],
                [0.43222603, 0.78825170, -0.62105948, 0.10097823, 0.81639361],
                [0.36671457, 0.24468753, -0.58248740, -0.74464536, -0.38901371],
            ]
        ]
    )
    input = flow.Tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    weight = np.array(
        [
            [
                [-0.29574063, -0.31176069, 0.17234495],
                [0.06092392, 0.30691007, -0.36685407],
            ],
            [
                [0.26149744, 0.07149458, 0.32097560],
                [0.18960869, -0.37148297, -0.13602243],
            ],
        ]
    )
    bias = np.array([-0.35048512, -0.00937920])
    m = nn.Conv1d(4, 2, 3, groups=2, stride=1, bias=True)
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m.bias = flow.nn.Parameter(flow.Tensor(bias))
    m = m.to(device)
    np_out = np.array(
        [[[-1.09048378, -0.49156523, 0.99150705], [0.01852397, 0.54882324, 0.31657016]]]
    )
    output = m(input)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
    output = output.sum()
    output.backward()
    np_grad = np.array(
        [
            [
                [-0.29574063, -0.60750133, -0.43515638, -0.13941574, 0.17234495],
                [0.06092392, 0.36783397, 0.00097990, -0.05994400, -0.36685407],
                [0.26149744, 0.33299202, 0.65396762, 0.39247018, 0.32097560],
                [0.18960869, -0.18187428, -0.31789672, -0.50750542, -0.13602243],
            ]
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestConv1d(flow.unittest.TestCase):
    def test_conv1d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_conv1d_bias_true,
            _test_conv1d_group_bias_true,
            _test_conv1d_group_large_out_bias_true,
            _test_conv1d_group_large_in_bias_true,
        ]
        arg_dict["device"] = ["cuda", "cpu"]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
