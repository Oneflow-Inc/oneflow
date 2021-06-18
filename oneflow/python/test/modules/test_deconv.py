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
    m = nn.ConvTranspose2d(1, 2, 3, stride=1, bias=False)
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
    np_grad = [
        [
            [
                [0.24731683731079102, 0.24731683731079102],
                [0.24731683731079102, 0.24731683731079102],
            ]
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


def _test_deconv_bias_true(test_case, device):
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
    bias = np.array([0.06456436216831207, -0.10852358490228653])
    m = nn.ConvTranspose2d(1, 2, 3, stride=1)
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m.bias = flow.nn.Parameter(flow.Tensor(bias))
    m = m.to(device)
    output = m(input)
    np_out = [
        [
            [
                [
                    0.0822228491306305,
                    -0.05448906123638153,
                    0.15560373663902283,
                    0.36409419775009155,
                ],
                [
                    0.07057422399520874,
                    0.30337145924568176,
                    -0.3120154142379761,
                    -0.19743980467319489,
                ],
                [
                    -0.16294024884700775,
                    0.188615083694458,
                    0.12139624357223511,
                    0.029504351317882538,
                ],
                [
                    0.11777572333812714,
                    0.17737999558448792,
                    0.08266734331846237,
                    0.06386750191450119,
                ],
            ],
            [
                [
                    -0.05242963880300522,
                    -0.3516995906829834,
                    -0.3787403702735901,
                    0.21595585346221924,
                ],
                [
                    0.15465816855430603,
                    -0.25121501088142395,
                    -0.027742713689804077,
                    -0.2504381537437439,
                ],
                [
                    0.028003744781017303,
                    -0.088503897190094,
                    -0.2181154191493988,
                    -0.139246866106987,
                ],
                [
                    -0.2703717350959778,
                    0.07788405567407608,
                    -0.09363573789596558,
                    -0.10918587446212769,
                ],
            ],
        ]
    ]

    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
    output = output.sum()
    output.backward()
    np_grad = [
        [
            [
                [0.24731683731079102, 0.24731683731079102],
                [0.24731683731079102, 0.24731683731079102],
            ]
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


def _test_deconv_group(test_case, device):
    np_arr = np.random.randn(1, 2, 2, 2)
    input = flow.Tensor(np_arr, dtype=flow.float32, device=flow.device("cuda"), requires_grad=True)
    m = nn.ConvTranspose2d(2, 2, 3, stride=1, groups=2, bias=False)
    m = m.to("cuda")
    output = m(input)
    print(np_arr.tolist())
    print(m.weight.numpy().tolist())
    print(output.numpy().tolist())
    



@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLess(flow.unittest.TestCase):
    def test_less(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_deconv_bias_false,
            _test_deconv_bias_true,
            _test_deconv_group,
        ]
        arg_dict["device"] = ["cuda", "cpu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
