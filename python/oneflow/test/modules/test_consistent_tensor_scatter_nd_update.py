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
import oneflow.unittest

from test_util import GenArgList
from oneflow.test_utils.automated_test_util import *


class TensorScatterNdUpdate(flow.nn.Graph):
    def __init__(self):
        super(TensorScatterNdUpdate, self).__init__()

    def build(self, origin, indices, update):
        return flow.tensor_scatter_nd_update(origin, indices, update)


def _test_global_tensor_scatter_nd_update(
    test_case, placement, sbp, check_graph=False
):
    origin = flow.tensor(np.arange(16), dtype=flow.float).to_global(
        flow.env.all_device_placement("cpu"), flow.sbp.broadcast
    )
    origin = origin.to_global(placement, sbp)
    indices = flow.tensor(
        np.array([[1], [6], [4], [3], [8], [10], [5], [7]]), dtype=flow.int
    ).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.hierarchy))]
    )
    update = flow.tensor(
        np.array([10.2, 5.1, 12.7, 5.4, 9.2, 3.1, 4.2, 5.2]), dtype=flow.float
    ).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.hierarchy))]
    )
    np_out = np.array(
        [
            0.0,
            10.2,
            2.0,
            5.4,
            12.7,
            4.2,
            5.1,
            5.2,
            9.2,
            9.0,
            3.1,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
        ]
    )
    if check_graph:
        tensor_scatter_nd_update = TensorScatterNdUpdate()
        output = tensor_scatter_nd_update(origin, indices, update)
    else:
        output = flow.tensor_scatter_nd_update(origin, indices, update)

    test_case.assertTrue(np.allclose(output.numpy(), np_out, 0.0001, 0.0001))


def _test_global_tensor_scatter_nd_update_t(
    test_case, placement, sbp, check_graph=False
):
    origin = flow.tensor(np.arange(32).reshape(8, 4), dtype=flow.float).to_global(
        flow.env.all_device_placement("cpu"), flow.sbp.broadcast
    )
    origin = origin.to_global(placement, sbp)
    indices = flow.tensor(np.array([[0], [4], [2]]), dtype=flow.int).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.hierarchy))]
    )
    update = flow.tensor(
        np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]), dtype=flow.float,
    ).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.hierarchy))]
    )
    np_out = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [4.0, 5.0, 6.0, 7.0],
            [3.0, 3.0, 3.0, 3.0],
            [12.0, 13.0, 14.0, 15.0],
            [2.0, 2.0, 2.0, 2.0],
            [20.0, 21.0, 22.0, 23.0],
            [24.0, 25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0, 31.0],
        ]
    )
    if check_graph:
        tensor_scatter_nd_update = TensorScatterNdUpdate()
        output = tensor_scatter_nd_update(origin, indices, update)
    else:
        output = flow.tensor_scatter_nd_update(origin, indices, update)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 0.0001, 0.0001))


def _test_eager_global_tensor_scatter_nd_update_backward(test_case, placement, sbp):
    origin = flow.tensor(
        np.arange(16), dtype=flow.float, requires_grad=True,
    ).to_global(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
    origin = origin.to_global(placement, sbp)

    indices = flow.tensor(
        np.array([[1], [6], [4], [3], [8], [10], [5], [7]]), dtype=flow.int
    ).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.hierarchy))]
    )

    of_update = flow.tensor(
        np.array([10.2, 5.1, 12.7, 5.4, 9.2, 3.1, 4.2, 5.2]),
        requires_grad=True,
        dtype=flow.float,
    ).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.hierarchy))]
    )
    np_out = np.array(
        [
            0.0,
            10.2,
            2.0,
            5.4,
            12.7,
            4.2,
            5.1,
            5.2,
            9.2,
            9.0,
            3.1,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
        ]
    )
    np_update_grad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    np_origin_grad = np.array(
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    output = flow.tensor_scatter_nd_update(origin, indices, of_update)
    out_sum = output.sum()
    out_sum.backward()
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(of_update.grad.numpy(), np_update_grad))
    test_case.assertTrue(np.allclose(origin.grad.numpy(), np_origin_grad))


class TestTensorScatterNdUpdate(flow.unittest.TestCase):
    @global_view
    def test_global_tensor_scatter_nd_update(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_tensor_scatter_nd_update(
                    test_case, placement, sbp, False
                )  # eager global
                _test_global_tensor_scatter_nd_update(
                    test_case, placement, sbp, True
                )  # nn graph

    @global_view
    def test_global_tensor_scatter_nd_update_t(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_tensor_scatter_nd_update_t(
                    test_case, placement, sbp, False
                )  # eager global
                _test_global_tensor_scatter_nd_update_t(
                    test_case, placement, sbp, True
                )  # nn graph

    @global_view
    def test_global_tensor_scatter_nd_update_backward(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_eager_global_tensor_scatter_nd_update_backward(
                    test_case, placement, sbp
                )


if __name__ == "__main__":
    unittest.main()
