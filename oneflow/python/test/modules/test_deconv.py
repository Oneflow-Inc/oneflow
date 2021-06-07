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


def _test_deconv_bias_false(test_case, device):
    np_arr = np.array(
        [
            [
                [
                    [0.2735021114349365, -1.3842310905456543],
                    [1.058540940284729, -0.03388553857803345],
                ]
            ]
        ]
    )
    input = flow.Tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    weight = np.array(
        [
            [
                [
                    [0.06456436216831207, -0.10852358490228653, -0.21638715267181396],
                    [-0.2279110550880432, 0.1476770043373108, 0.19457484781742096],
                    [0.05026858672499657, 0.10818571597337723, 0.02056501805782318],
                ],
                [
                    [0.205095112323761, 0.1488947868347168, -0.2344113141298294],
                    [0.1684819906949997, -0.21986986696720123, 0.1082606166601181],
                    [-0.1528974026441574, 0.17120417952537537, 0.01954500749707222],
                ],
            ]
        ]
    )
    m = nn.ConvTranspose2d(1, 2, 3, stride=1)
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m = m.to(device)
    output = m(input)
    np_out = np.array(
        [
            [
                [
                    [
                        0.01765848882496357,
                        -0.1190534234046936,
                        0.09103937447071075,
                        0.2995298206806183,
                    ],
                    [
                        0.006009865552186966,
                        0.2388070970773697,
                        -0.37657976150512695,
                        -0.26200416684150696,
                    ],
                    [
                        -0.22750461101531982,
                        0.12405071407556534,
                        0.056831881403923035,
                        -0.035060010850429535,
                    ],
                    [
                        0.053211357444524765,
                        0.11281562596559525,
                        0.0181029811501503,
                        -0.0006968567031435668,
                    ],
                ],
                [
                    [
                        0.05609394609928131,
                        -0.24317599833011627,
                        -0.27021679282188416,
                        0.32447943091392517,
                    ],
                    [
                        0.26318174600601196,
                        -0.14269141852855682,
                        0.08078087121248245,
                        -0.14191456139087677,
                    ],
                    [
                        0.13652732968330383,
                        0.020019691437482834,
                        -0.10959184169769287,
                        -0.03072327747941017,
                    ],
                    [
                        -0.16184815764427185,
                        0.1864076405763626,
                        0.014887845143675804,
                        -0.0006622931105084717,
                    ],
                ],
            ]
        ]
    )
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
    output = output.sum()
    output.backward()
    np_grad = [[[[0.24731683731079102, 0.24731683731079102], [0.24731683731079102, 0.24731683731079102]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLess(flow.unittest.TestCase):
    def test_less(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_deconv_bias_false,
        ]
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
