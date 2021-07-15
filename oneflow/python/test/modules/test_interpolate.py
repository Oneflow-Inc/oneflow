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


def _test_interpolate_linear_1d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 5).reshape((1, 1, 4)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    m = flow.nn.functional.interpolate(scale_factor=2.0, mode="linear")
    of_out = m(input)
    np_out = [[[1.0, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.0]]]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[2.0, 2.0, 2.0, 2.0]]]
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 1e-4, 1e-4))

    input.grad = None
    m = flow.nn.functional.interpolate(
        scale_factor=2.0, mode="linear", align_corners=True
    )
    of_out = m(input)
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
    m = flow.nn.functional.interpolate(scale_factor=2.0, mode="nearest")
    of_out = m(input)
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
    m = flow.nn.functional.interpolate(scale_factor=2.0, mode="nearest")
    of_out = m(input)
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
    m = flow.nn.functional.interpolate(scale_factor=2.0, mode="nearest")
    of_out = m(input)
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
    m = flow.nn.functional.interpolate(scale_factor=2.0, mode="bilinear")
    of_out = m(input)
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
    m = flow.nn.functional.interpolate(scale_factor=2.0, mode="bicubic")
    of_out = m(input)
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


def _test_interpolate_trilinear_3d(test_case, device):
    input = flow.Tensor(
        np.arange(1, 9).reshape((1, 1, 2, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    m = flow.nn.functional.interpolate(scale_factor=2.0, mode="trilinear")
    of_out = m(input)
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
    m = flow.nn.functional.interpolate(
        scale_factor=2.0, mode="trilinear", align_corners=True
    )
    of_out = m(input)
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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestUpsample2d(flow.unittest.TestCase):
    def test_upsample2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_interpolate_linear_1d,
            _test_interpolate_nearest_1d,
            _test_interpolate_nearest_2d,
            _test_interpolate_nearest_3d,
            _test_interpolate_bilinear_2d,
            _test_interpolate_bicubic_2d,
            _test_interpolate_trilinear_3d,
            _test_interpolate_trilinear_3d_align_corners,
        ]
        arg_dict["device"] = [
            "cpu",
            # "cuda",
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
