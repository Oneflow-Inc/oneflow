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

import oneflow as flow
from test_util import GenArgList


def _test_interpolate_linear_1d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 5).reshape((1, 1, 4)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="linear")
    np_out = [[[1.0, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.0]]]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[2.0, 2.0, 2.0, 2.0]]]
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 1e-4, 1e-4))

    input.grad = None
    of_out = flow.nn.functional.interpolate(
        input, scale_factor=2.0, mode="linear", align_corners=True
    )
    np_out = [
        [
            [
                1.0,
                1.4285714626312256,
                1.8571429252624512,
                2.2857141494750977,
                2.7142856121063232,
                3.142857074737549,
                3.5714285373687744,
                4.0,
            ]
        ]
    ]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [
            [
                1.7142856121063232,
                2.2857141494750977,
                2.2857143878936768,
                1.7142856121063232,
            ]
        ]
    ]
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 1e-4, 1e-4))


def _test_interpolate_nearest_1d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 5).reshape((1, 1, 4)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="nearest")
    np_out = [[[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]]]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[2.0, 2.0, 2.0, 2.0]]]
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 1e-4, 1e-4))


def _test_interpolate_nearest_2d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="nearest")
    np_out = np.array(
        [
            [
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[4.0, 4.0], [4.0, 4.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate_nearest_3d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 9).reshape((1, 1, 2, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="nearest")
    np_out = np.array(
        [
            [
                [
                    [
                        [1.0, 1.0, 2.0, 2.0],
                        [1.0, 1.0, 2.0, 2.0],
                        [3.0, 3.0, 4.0, 4.0],
                        [3.0, 3.0, 4.0, 4.0],
                    ],
                    [
                        [1.0, 1.0, 2.0, 2.0],
                        [1.0, 1.0, 2.0, 2.0],
                        [3.0, 3.0, 4.0, 4.0],
                        [3.0, 3.0, 4.0, 4.0],
                    ],
                    [
                        [5.0, 5.0, 6.0, 6.0],
                        [5.0, 5.0, 6.0, 6.0],
                        [7.0, 7.0, 8.0, 8.0],
                        [7.0, 7.0, 8.0, 8.0],
                    ],
                    [
                        [5.0, 5.0, 6.0, 6.0],
                        [5.0, 5.0, 6.0, 6.0],
                        [7.0, 7.0, 8.0, 8.0],
                        [7.0, 7.0, 8.0, 8.0],
                    ],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[[8.0, 8.0], [8.0, 8.0]], [[8.0, 8.0], [8.0, 8.0]]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate_bilinear_2d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="bilinear")
    np_out = np.array(
        [
            [
                [
                    [1.0, 1.25, 1.75, 2.0],
                    [1.5, 1.75, 2.25, 2.5],
                    [2.5, 2.75, 3.25, 3.5],
                    [3.0, 3.25, 3.75, 4.0],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[4.0, 4.0], [4.0, 4.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate_bicubic_2d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)).astype(np.float32),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="bicubic")
    np_out = np.array(
        [
            [
                [
                    [0.68359375, 1.015625, 1.5625, 1.89453125],
                    [1.34765625, 1.6796875, 2.2265625, 2.55859375],
                    [2.44140625, 2.7734375, 3.3203125, 3.65234375],
                    [3.10546875, 3.4375, 3.984375, 4.31640625],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[4.0, 4.0], [4.0, 4.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate_bicubic_same_dim_2d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)).astype(np.float32),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=1.0, mode="bicubic")
    np_out = [[[[1.0, 2.0], [3.0, 4.0]]]]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[1.0, 1.0], [1.0, 1.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate_trilinear_3d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 9).reshape((1, 1, 2, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="trilinear")
    np_out = np.array(
        [
            [
                [
                    [
                        [1.0, 1.25, 1.75, 2.0],
                        [1.5, 1.75, 2.25, 2.5],
                        [2.5, 2.75, 3.25, 3.5],
                        [3.0, 3.25, 3.75, 4.0],
                    ],
                    [
                        [2.0, 2.25, 2.75, 3.0],
                        [2.5, 2.75, 3.25, 3.5],
                        [3.5, 3.75, 4.25, 4.5],
                        [4.0, 4.25, 4.75, 5.0],
                    ],
                    [
                        [4.0, 4.25, 4.75, 5.0],
                        [4.5, 4.75, 5.25, 5.5],
                        [5.5, 5.75, 6.25, 6.5],
                        [6.0, 6.25, 6.75, 7.0],
                    ],
                    [
                        [5.0, 5.25, 5.75, 6.0],
                        [5.5, 5.75, 6.25, 6.5],
                        [6.5, 6.75, 7.25, 7.5],
                        [7.0, 7.25, 7.75, 8.0],
                    ],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[[8.0, 8.0], [8.0, 8.0]], [[8.0, 8.0], [8.0, 8.0]]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate_trilinear_3d_align_corners(test_case, device):
    input = flow.Tensor(
        np.arange(1, 9).reshape((1, 1, 2, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(
        input, scale_factor=2.0, mode="trilinear", align_corners=True
    )
    np_out = np.array(
        [
            [
                [
                    [
                        [1.0, 1.3333332538604736, 1.6666667461395264, 2.0],
                        [
                            1.6666666269302368,
                            2.0,
                            2.3333334922790527,
                            2.6666665077209473,
                        ],
                        [
                            2.3333332538604736,
                            2.6666665077209473,
                            3.0,
                            3.3333334922790527,
                        ],
                        [3.0, 3.3333332538604736, 3.6666667461395264, 4.0],
                    ],
                    [
                        [
                            2.3333334922790527,
                            2.6666665077209473,
                            3.0,
                            3.3333332538604736,
                        ],
                        [3.0, 3.3333330154418945, 3.6666665077209473, 4.0],
                        [
                            3.6666665077209473,
                            4.0,
                            4.333333492279053,
                            4.6666669845581055,
                        ],
                        [4.333333492279053, 4.666666030883789, 5.0, 5.3333330154418945],
                    ],
                    [
                        [3.6666667461395264, 4.0, 4.333333492279053, 4.666666507720947],
                        [4.333333492279053, 4.666666507720947, 5.0, 5.3333330154418945],
                        [5.0, 5.333333492279053, 5.6666669845581055, 6.0],
                        [
                            5.6666669845581055,
                            6.0,
                            6.333333492279053,
                            6.6666669845581055,
                        ],
                    ],
                    [
                        [5.0, 5.3333330154418945, 5.666666507720947, 6.0],
                        [
                            5.666666507720947,
                            5.999999523162842,
                            6.3333330154418945,
                            6.666666507720947,
                        ],
                        [6.333333492279053, 6.666666030883789, 7.0, 7.333333492279053],
                        [7.0, 7.3333330154418945, 7.6666669845581055, 8.0],
                    ],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [
            [
                [[7.999999523162842, 8.0], [7.999999523162842, 8.0]],
                [[8.0, 8.0], [8.0, 8.0]],
            ]
        ]
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate_area_1d(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        0.05580734834074974,
                        -0.6875145435333252,
                        -1.654430866241455,
                        -0.6225992441177368,
                        0.10183599591255188,
                        0.05019790679216385,
                        -1.2537643909454346,
                        0.14907236397266388,
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out_1 = flow.nn.functional.interpolate(input, size=4, mode="area")
    of_out_2 = flow.nn.functional.interpolate(input, scale_factor=0.5, mode="area")
    np_out = np.array(
        [
            [
                [
                    -0.31585359573364258,
                    -1.13851499557495117,
                    0.07601694762706757,
                    -0.55234599113464355,
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out, 1e-5, 1e-5))

    of_out_1 = of_out_1.sum()
    of_out_1.backward()
    np_grad = np.array([[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]])
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate_area_2d(test_case, device):
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
    of_out_1 = flow.nn.functional.interpolate(input, size=(2, 2), mode="area")
    of_out_2 = flow.nn.functional.interpolate(input, scale_factor=0.5, mode="area")
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
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out, 1e-5, 1e-5))

    of_out_1 = of_out_1.sum()
    of_out_1.backward()
    np_grad = np.array(
        [
            [
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate_area_3d(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [
                            [
                                -1.07757179960088489,
                                -0.78045388903658375,
                                -1.26275387521194427,
                                0.99935071451204771,
                            ],
                            [
                                2.02225324891575164,
                                1.10345137769946500,
                                -0.43773247548795780,
                                1.89049181058751703,
                            ],
                            [
                                -0.55938618990646538,
                                -0.49495202415265188,
                                -0.18536721363519787,
                                -0.60989698667757719,
                            ],
                            [
                                -1.65362152601718160,
                                -1.03925835404367861,
                                0.36867765976139671,
                                -0.53568828349518050,
                            ],
                        ],
                        [
                            [
                                -1.26179006644499525,
                                -1.43909210916315322,
                                0.20654399652431357,
                                0.81864721019067133,
                            ],
                            [
                                -0.30333788634000142,
                                -0.81732697640762930,
                                -0.37675150976256139,
                                -0.11021655039337777,
                            ],
                            [
                                -0.22977043608192885,
                                1.27171963666499055,
                                -0.47908512978782908,
                                -1.44953694047278558,
                            ],
                            [
                                -1.28020932869777826,
                                -0.11184514806663474,
                                1.70221670872109843,
                                -1.73548372877253554,
                            ],
                        ],
                        [
                            [
                                2.47064979917736061,
                                -0.65497026319732976,
                                -0.93181070795716758,
                                1.46529042716824276,
                            ],
                            [
                                1.14198642343413970,
                                1.38990908108600797,
                                0.96578419005255678,
                                -0.85631142649766190,
                            ],
                            [
                                0.19515087084250754,
                                -0.37808457398571094,
                                0.29386253984961830,
                                0.92799305103533269,
                            ],
                            [
                                -0.93741182779940069,
                                0.33418317304524309,
                                -0.27925427653038332,
                                0.38029090707066726,
                            ],
                        ],
                        [
                            [
                                0.59186866597360410,
                                -0.78706310899389020,
                                -0.95343448742453918,
                                0.31341612954718795,
                            ],
                            [
                                0.75090294441452277,
                                -0.92992883985623231,
                                -0.73430540527824761,
                                -0.88064815906966942,
                            ],
                            [
                                -0.47078530163539850,
                                0.12253641652645629,
                                0.50880220398328457,
                                0.52039178932756203,
                            ],
                            [
                                -0.08613006511636320,
                                0.30291348404866386,
                                -0.62685658736801231,
                                -0.27469204305759976,
                            ],
                        ],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out_1 = flow.nn.functional.interpolate(input, size=(2, 2, 2), mode="area")
    of_out_2 = flow.nn.functional.interpolate(input, scale_factor=0.5, mode="area")
    np_out = np.array(
        [
            [
                [
                    [
                        [-0.31923351254725391, 0.21594741511983859],
                        [-0.51216542128766618, -0.36552048929482639],
                    ],
                    [
                        [0.49666933775477279, -0.20150242993241230],
                        [-0.11470347800925032, 0.18131719803880864],
                    ],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out, 1e-5, 1e-5))

    of_out_1 = of_out_1.sum()
    of_out_1.backward()
    np_grad = np.array(
        [
            [
                [
                    [
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                    ],
                    [
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                    ],
                    [
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                    ],
                    [
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                    ],
                ]
            ]
        ]
    )

    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@flow.unittest.skip_unless_1n1d()
class TestInterpolate(flow.unittest.TestCase):
    def test_interpolate(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_interpolate_linear_1d,
            _test_interpolate_nearest_1d,
            _test_interpolate_nearest_2d,
            _test_interpolate_nearest_3d,
            _test_interpolate_bilinear_2d,
            _test_interpolate_bicubic_2d,
            _test_interpolate_bicubic_same_dim_2d,
            _test_interpolate_trilinear_3d,
            _test_interpolate_trilinear_3d_align_corners,
            _test_interpolate_area_1d,
            _test_interpolate_area_2d,
            _test_interpolate_area_3d,
        ]
        arg_dict["device"] = [
            "cpu",
            "cuda",
        ]
        for arg in GenArgList(arg_dict):
            for i in range(100):
                arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
