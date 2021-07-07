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


from test_util import GenArgList


def _test_adaptive_avgpool2d_forward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [
                            0.10039155930280685,
                            0.04879157617688179,
                            -1.0515470504760742,
                            0.9466001987457275,
                        ],
                        [
                            0.45375481247901917,
                            0.23611211776733398,
                            1.343685269355774,
                            0.3979687988758087,
                        ],
                        [
                            0.05580734834074974,
                            -0.6875145435333252,
                            -1.654430866241455,
                            -0.6225992441177368,
                        ],
                        [
                            0.10183599591255188,
                            0.05019790679216385,
                            -1.2537643909454346,
                            0.14907236397266388,
                        ],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
    )
    m = flow.nn.AdaptiveAvgPool2d((2, 2))
    m.to(device)
    of_out = m(input)
    np_out = np.array(
        [
            [
                [
                    [0.20976251363754272, 0.4091767966747284],
                    [-0.1199183315038681, -0.8454304933547974],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_adaptive_avgpool2d_backward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [
                            0.10039155930280685,
                            0.04879157617688179,
                            -1.0515470504760742,
                            0.9466001987457275,
                        ],
                        [
                            0.45375481247901917,
                            0.23611211776733398,
                            1.343685269355774,
                            0.3979687988758087,
                        ],
                        [
                            0.05580734834074974,
                            -0.6875145435333252,
                            -1.654430866241455,
                            -0.6225992441177368,
                        ],
                        [
                            0.10183599591255188,
                            0.05019790679216385,
                            -1.2537643909454346,
                            0.14907236397266388,
                        ],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    m = flow.nn.AdaptiveAvgPool2d((2, 2))
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ]
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_adaptive_avgpool2d_hw_forward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [0.28242185711860657, -0.7742040753364563, -0.5439430475234985],
                        [-0.1706847995519638, 0.0430854931473732, 0.34247592091560364],
                        [-1.036131501197815, -1.033642292022705, 0.3455536365509033],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
    )
    m = flow.nn.AdaptiveAvgPool2d((1, 2))
    m.to(device)
    of_out = m(input)
    np_out = np.array([[[[-0.4481925666332245, -0.27011242508888245]]]])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_adaptive_avgpool2d_hw_backward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [0.28242185711860657, -0.7742040753364563, -0.5439430475234985],
                        [-0.1706847995519638, 0.0430854931473732, 0.34247592091560364],
                        [-1.036131501197815, -1.033642292022705, 0.3455536365509033],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    m = flow.nn.AdaptiveAvgPool2d((1, 2))
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [
            [
                [0.1666666716337204, 0.3333333432674408, 0.1666666716337204],
                [0.1666666716337204, 0.3333333432674408, 0.1666666716337204],
                [0.1666666716337204, 0.3333333432674408, 0.1666666716337204],
            ]
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAdaptiveAvgPool2d(flow.unittest.TestCase):
    def test_adaptive_avgpool2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_adaptive_avgpool2d_forward,
            _test_adaptive_avgpool2d_backward,
            _test_adaptive_avgpool2d_hw_forward,
            _test_adaptive_avgpool2d_hw_backward,
        ]
        arg_dict["device"] = [
            "cpu",
            "cuda",
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
