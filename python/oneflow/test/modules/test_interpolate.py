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


def _test_interpolate_linear_1d(test_case, device):
    input = flow.tensor(
        np.arange(1, 5).reshape((1, 1, 4)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="linear")
    np_out = [[[1.0, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.0]]]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[2.0, 2.0, 2.0, 2.0]]]
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 0.0001, 0.0001))
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
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
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 0.0001, 0.0001))


def _test_interpolate_nearest_1d(test_case, device):
    input = flow.tensor(
        np.arange(1, 5).reshape((1, 1, 4)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="nearest")
    np_out = [[[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]]]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[2.0, 2.0, 2.0, 2.0]]]
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 0.0001, 0.0001))


def _test_interpolate_nearest_2d(test_case, device):
    input = flow.tensor(
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[4.0, 4.0], [4.0, 4.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_nearest_3d(test_case, device):
    input = flow.tensor(
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[[8.0, 8.0], [8.0, 8.0]], [[8.0, 8.0], [8.0, 8.0]]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_bilinear_2d(test_case, device):
    input = flow.tensor(
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[4.0, 4.0], [4.0, 4.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_bicubic_2d(test_case, device):
    input = flow.tensor(
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[4.0, 4.0], [4.0, 4.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_bicubic_same_dim_2d(test_case, device):
    input = flow.tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)).astype(np.float32),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    of_out = flow.nn.functional.interpolate(input, scale_factor=1.0, mode="bicubic")
    np_out = [[[[1.0, 2.0], [3.0, 4.0]]]]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[1.0, 1.0], [1.0, 1.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_trilinear_3d(test_case, device):
    input = flow.tensor(
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[[8.0, 8.0], [8.0, 8.0]], [[8.0, 8.0], [8.0, 8.0]]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_trilinear_3d_align_corners(test_case, device):
    input = flow.tensor(
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
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
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_area_1d(test_case, device):
    input = flow.tensor(
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
                    -0.3158535957336426,
                    -1.1385149955749512,
                    0.07601694762706757,
                    -0.5523459911346436,
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out, 1e-05, 1e-05))
    of_out_1 = of_out_1.sum()
    of_out_1.backward()
    np_grad = np.array([[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]])
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_area_2d(test_case, device):
    input = flow.tensor(
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
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out, 1e-05, 1e-05))
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
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_area_3d(test_case, device):
    input = flow.tensor(
        np.array(
            [
                [
                    [
                        [
                            [
                                -1.077571799600885,
                                -0.7804538890365837,
                                -1.2627538752119443,
                                0.9993507145120477,
                            ],
                            [
                                2.0222532489157516,
                                1.103451377699465,
                                -0.4377324754879578,
                                1.890491810587517,
                            ],
                            [
                                -0.5593861899064654,
                                -0.4949520241526519,
                                -0.18536721363519787,
                                -0.6098969866775772,
                            ],
                            [
                                -1.6536215260171816,
                                -1.0392583540436786,
                                0.3686776597613967,
                                -0.5356882834951805,
                            ],
                        ],
                        [
                            [
                                -1.2617900664449953,
                                -1.4390921091631532,
                                0.20654399652431357,
                                0.8186472101906713,
                            ],
                            [
                                -0.3033378863400014,
                                -0.8173269764076293,
                                -0.3767515097625614,
                                -0.11021655039337777,
                            ],
                            [
                                -0.22977043608192885,
                                1.2717196366649905,
                                -0.4790851297878291,
                                -1.4495369404727856,
                            ],
                            [
                                -1.2802093286977783,
                                -0.11184514806663474,
                                1.7022167087210984,
                                -1.7354837287725355,
                            ],
                        ],
                        [
                            [
                                2.4706497991773606,
                                -0.6549702631973298,
                                -0.9318107079571676,
                                1.4652904271682428,
                            ],
                            [
                                1.1419864234341397,
                                1.389909081086008,
                                0.9657841900525568,
                                -0.8563114264976619,
                            ],
                            [
                                0.19515087084250754,
                                -0.37808457398571094,
                                0.2938625398496183,
                                0.9279930510353327,
                            ],
                            [
                                -0.9374118277994007,
                                0.3341831730452431,
                                -0.2792542765303833,
                                0.38029090707066726,
                            ],
                        ],
                        [
                            [
                                0.5918686659736041,
                                -0.7870631089938902,
                                -0.9534344874245392,
                                0.31341612954718795,
                            ],
                            [
                                0.7509029444145228,
                                -0.9299288398562323,
                                -0.7343054052782476,
                                -0.8806481590696694,
                            ],
                            [
                                -0.4707853016353985,
                                0.12253641652645629,
                                0.5088022039832846,
                                0.520391789327562,
                            ],
                            [
                                -0.0861300651163632,
                                0.30291348404866386,
                                -0.6268565873680123,
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
                        [-0.3192335125472539, 0.2159474151198386],
                        [-0.5121654212876662, -0.3655204892948264],
                    ],
                    [
                        [0.4966693377547728, -0.2015024299324123],
                        [-0.11470347800925032, 0.18131719803880864],
                    ],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out, 1e-05, 1e-05))
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
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_output_size_arg_with_scalar(test_case, device):
    mode = "bicubic"
    x = flow.Tensor(8, 32, 64).to(device)

    window = 16
    t = x.shape[2]
    x = x[:, None]

    np_center = np.random.randint(window, t - window, (1,))[0]
    np_warped = np.random.randint(np_center - window, np_center + window, (1,))[0] + 1

    center = flow.tensor(np_center)
    warped = flow.tensor(np_warped)

    res = flow.nn.functional.interpolate(
        x[:, :, :center], (warped, x.shape[3]), mode=mode, align_corners=False
    )
    test_case.assertTrue(np.array_equal(res.size()[0], 8))
    test_case.assertTrue(np.array_equal(res.size()[1], 1))


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
            _test_interpolate_output_size_arg_with_scalar,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            for i in range(100):
                arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
