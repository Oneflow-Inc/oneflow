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

from oneflow.test_utils.automated_test_util import *


class TensorScatterNdUpdate(flow.nn.Graph):
    def __init__(self):
        super(TensorScatterNdUpdate, self).__init__()

    def build(self, origin, indices, update):
        return flow.tensor_scatter_nd_update(origin, indices, update)


def _test_global_tensor_scatter_nd_update(test_case, placement, sbp, check_graph=True):
    origin = random_tensor(1, 16, requires_grad=False).to_global(placement, sbp)
    indices = choice_tensor(16, (8, 1), replace=False).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.ranks.shape))]
    )
    update = random_tensor(1, 8, requires_grad=False).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.ranks.shape))]
    )

    np_origin = origin.oneflow.numpy()
    np_indices = indices.oneflow.numpy().reshape(8)
    np_update = update.oneflow.numpy()

    if check_graph:
        tensor_scatter_nd_update = TensorScatterNdUpdate()
        output = tensor_scatter_nd_update(
            origin.oneflow, indices.oneflow, update.oneflow
        )
    else:
        output = flow.tensor_scatter_nd_update(
            origin.oneflow, indices.oneflow, update.oneflow
        )

    np_origin[np_indices] = np_update

    test_case.assertTrue(np.allclose(output.numpy(), np_origin, 0.0001, 0.0001))


def _test_global_tensor_scatter_nd_update_t(
    test_case, placement, sbp, check_graph=True
):

    origin = random_tensor(2, 16, 4, requires_grad=False).to_global(placement, sbp)
    indices = choice_tensor(16, (8, 1), replace=False).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.ranks.shape))]
    )
    update = random_tensor(2, 8, 4, requires_grad=False).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.ranks.shape))]
    )

    np_origin = origin.oneflow.numpy()
    np_indices = indices.oneflow.numpy().reshape(8)
    np_update = update.oneflow.numpy()

    if check_graph:
        tensor_scatter_nd_update = TensorScatterNdUpdate()
        output = tensor_scatter_nd_update(
            origin.oneflow, indices.oneflow, update.oneflow
        )
    else:
        output = flow.tensor_scatter_nd_update(
            origin.oneflow, indices.oneflow, update.oneflow
        )

    np_origin[np_indices] = np_update

    test_case.assertTrue(np.allclose(output.numpy(), np_origin, 0.0001, 0.0001))


def _test_eager_global_tensor_scatter_nd_update_backward(test_case, placement, sbp):
    origin = random_tensor(1, 16,).to_global(placement, sbp)
    origin.retain_grad()
    indices = choice_tensor(16, (8, 1), replace=False).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.ranks.shape))]
    )
    update = random_tensor(1, 8).to_global(
        placement, [flow.sbp.broadcast for _ in range(len(placement.ranks.shape))]
    )
    update.retain_grad()

    np_origin = origin.oneflow.numpy()
    np_indices = indices.oneflow.numpy().reshape(8)
    np_update = update.oneflow.numpy()

    np_update_grad = np.ones(8)
    np_origin_grad = np.ones(16)
    np_origin_grad[np_indices] = np.zeros(8)

    output = flow.tensor_scatter_nd_update(
        origin.oneflow, indices.oneflow, update.oneflow
    )
    out_sum = output.sum()
    out_sum.backward()

    np_origin[np_indices] = np_update

    test_case.assertTrue(np.allclose(output.numpy(), np_origin, 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(update.oneflow.grad.numpy(), np_update_grad))
    test_case.assertTrue(np.allclose(origin.oneflow.grad.numpy(), np_origin_grad))


class TestTensorScatterNdUpdate(flow.unittest.TestCase):
    @globaltest
    def test_global_tensor_scatter_nd_update(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_tensor_scatter_nd_update(
                    test_case, placement, sbp, False
                )  # eager global
                # skip lazy test
                # _test_global_tensor_scatter_nd_update(
                #     test_case, placement, sbp, True
                # )  # nn graph

    @globaltest
    def test_global_tensor_scatter_nd_update_t(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_tensor_scatter_nd_update_t(
                    test_case, placement, sbp, False
                )  # eager global
                # skip lazy test
                # _test_global_tensor_scatter_nd_update_t(
                #     test_case, placement, sbp, True
                # )  # nn graph

    @globaltest
    def test_global_tensor_scatter_nd_update_backward(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_eager_global_tensor_scatter_nd_update_backward(
                    test_case, placement, sbp
                )


if __name__ == "__main__":
    unittest.main()
