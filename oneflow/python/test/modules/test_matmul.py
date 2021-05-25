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
from collections import OrderedDict

import unittest
import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList


def _test_matmul(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(6, 5), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.matmul(input1, input2)
    np_out = np.matmul(input1.numpy(), input2.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_broadcast_matmul(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(3, 4, 5), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(5, 6), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.matmul(input1, input2)
    np_out = np.matmul(input1.numpy(), input2.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_batch_matmul(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(10, 3, 4), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(10, 4, 5), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.matmul(input1, input2)
    np_out = np.matmul(input1.numpy(), input2.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_matmul_backward(test_case, device):
    input1 = flow.Tensor(
        [
            [
                -0.36023932695388794,
                0.5571867227554321,
                -1.4987696409225464,
                -0.9674592018127441,
                0.021076146513223648,
                2.9180469512939453,
            ],
            [
                -0.29169487953186035,
                0.2978641390800476,
                0.8198832273483276,
                -0.3385652005672455,
                -2.9260432720184326,
                0.22528153657913208,
            ],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input2 = flow.Tensor(
        [
            [
                -0.5270200371742249,
                -0.4325239062309265,
                -0.33396217226982117,
                1.2983192205429077,
                -0.463693231344223,
            ],
            [
                1.893467903137207,
                -1.0874812602996826,
                0.7068315744400024,
                -0.23532593250274658,
                -0.011510828509926796,
            ],
            [
                -0.5477776527404785,
                -0.0381619855761528,
                0.03451986983418465,
                -0.8248650431632996,
                -1.8885509967803955,
            ],
            [
                -1.0034432411193848,
                0.5428839921951294,
                -0.7785694599151611,
                -0.4489346146583557,
                1.780846118927002,
            ],
            [
                0.9378347396850586,
                -0.38816362619400024,
                0.8186876177787781,
                -0.9630932807922363,
                -0.11487948149442673,
            ],
            [
                -0.12073716521263123,
                2.181835174560547,
                0.5511962175369263,
                -1.294308066368103,
                -0.7765272855758667,
            ],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    print("input1.shape >>>>>>>>>>>>>>>>>>>>>> \n", input1.shape)
    print("input2.shape >>>>>>>>>>>>>>>>>>>>>> \n", input2.shape)
    of_out = flow.matmul(input1, input2)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [
            -0.45888009667396545,
            1.2659813165664673,
            -3.264835834503174,
            0.09278273582458496,
            0.2903860807418823,
            0.5414588451385498,
        ],
        [
            -0.45888009667396545,
            1.2659813165664673,
            -3.264835834503174,
            0.09278273582458496,
            0.2903860807418823,
            0.5414588451385498,
        ],
    ]
    print("input1.grad.numpy() >>>>>>>>>>>>>>>>>>>>>>>> \n", input1.grad.numpy())
    test_case.assertTrue(
        np.allclose(input1.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_matmul_backward_x_grad(test_case, device):
    input1 = flow.Tensor(
        [
            [-1.8604081869125366, -2.0019688606262207],
            [1.0511547327041626, -2.263841390609741],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input2 = flow.Tensor(
        [
            [-0.13973912596702576, 0.8478717803955078],
            [-0.2144828885793686, -1.7145386934280396],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=False,
    )
    of_out = flow.matmul(input1, input2)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [0.7081326246261597, -1.9290215969085693],
        [0.7081326246261597, -1.9290215969085693],
    ]
    test_case.assertTrue(
        np.allclose(input1.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_matmul_backward_y_grad(test_case, device):
    input1 = flow.Tensor(
        [
            [-1.8604081869125366, -2.0019688606262207],
            [1.0511547327041626, -2.263841390609741],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=False,
    )
    input2 = flow.Tensor(
        [
            [-0.13973912596702576, 0.8478717803955078],
            [-0.2144828885793686, -1.7145386934280396],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.matmul(input1, input2)
    of_out = of_out.sum()
    of_out.backward()
    print(input2.grad.numpy().tolist())
    np_grad = [
        [-0.809253454208374, -0.809253454208374],
        [-4.265810012817383, -4.265810012817383],
    ]
    test_case.assertTrue(
        np.allclose(input2.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_broadcast_matmul_backward(test_case, device):
    input1 = flow.Tensor(
        [
            [
                [0.5893293023109436, -0.0376124233007431, 0.7791574001312256],
                [1.1614371538162231, 0.009700910188257694, 0.7281601428985596],
            ],
            [
                [-0.27213698625564575, 0.7058051824569702, -0.4643424451351166],
                [2.2279646396636963, 0.05870082601904869, -0.18335142731666565],
            ],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input2 = flow.Tensor(
        [
            [0.25825661420822144, -0.4875393807888031],
            [-0.040459781885147095, -0.3713535666465759],
            [-1.633512258529663, -2.0034799575805664],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.matmul(input1, input2)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [
            [-0.22928276658058167, -0.411813348531723, -3.6369922161102295],
            [-0.22928276658058167, -0.411813348531723, -3.6369922161102295],
        ],
        [
            [-0.22928276658058167, -0.411813348531723, -3.6369922161102295],
            [-0.22928276658058167, -0.411813348531723, -3.6369922161102295],
        ],
    ]
    test_case.assertTrue(
        np.allclose(input1.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


def _test_batch_matmul_backward(test_case, device):
    input1 = flow.Tensor(
        [
            [
                [-0.0036776792258024216, 1.9946473836898804, -0.423959881067276],
                [1.0892143249511719, 0.04005361348390579, -0.27883127331733704],
            ],
            [
                [-0.970306396484375, 0.017771577462553978, 0.019596196711063385],
                [0.27402883768081665, -0.8192587494850159, -0.3135920464992523],
            ],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input2 = flow.Tensor(
        [
            [
                [1.118346929550171, -0.930071234703064],
                [1.1238232851028442, 1.373764157295227],
                [0.17178462445735931, -1.1010534763336182],
            ],
            [
                [0.6694859862327576, 0.9250285029411316],
                [-1.0835869312286377, 0.4192655086517334],
                [1.2616937160491943, 0.33809131383895874],
            ],
        ],
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.matmul(input1, input2)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [
            [0.18827569484710693, 2.4975874423980713, -0.9292688369750977],
            [0.18827569484710693, 2.4975874423980713, -0.9292688369750977],
        ],
        [
            [1.5945144891738892, -0.6643214225769043, 1.5997850894927979],
            [1.5945144891738892, -0.6643214225769043, 1.5997850894927979],
        ],
    ]
    test_case.assertTrue(
        np.allclose(input1.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestModule(flow.unittest.TestCase):
    def test_matmul(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_matmul,
            _test_broadcast_matmul,
            _test_batch_matmul,
            _test_matmul_backward,
            _test_matmul_backward_x_grad,
            _test_matmul_backward_y_grad,
            _test_broadcast_matmul_backward,
            _test_batch_matmul_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
