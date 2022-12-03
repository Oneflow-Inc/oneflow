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

import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


def _test_scatter_nd(test_case, placement, sbp):
    indices = (
        flow.tensor(np.array([[1], [6], [4]]), dtype=flow.int)
        .to_global(flow.placement.all("cpu"), [flow.sbp.broadcast,])
        .to_global(placement, sbp)
    )
    update = (
        flow.tensor(np.array([10.2, 5.1, 12.7]), dtype=flow.float)
        .to_global(flow.placement.all("cpu"), [flow.sbp.broadcast,])
        .to_global(placement, sbp)
        .requires_grad_()
    )
    output = flow.scatter_nd(indices, update, [8])

    # forward
    of_local = output.to_global(
        flow.placement.all("cpu"), [flow.sbp.broadcast,]
    ).to_local()
    np_out = np.array([0.0, 10.2, 0.0, 0.0, 12.7, 0.0, 5.1, 0.0])
    test_case.assertTrue(np.allclose(of_local.numpy(), np_out, 1e-4, 1e-4))

    # backward
    output.sum().backward()
    of_grad_local = update.grad.to_global(
        flow.placement.all("cpu"), [flow.sbp.broadcast,]
    ).to_local()
    test_case.assertTrue(np.allclose(of_grad_local.numpy(), np.ones((3)), 1e-4, 1e-4))


class TestScatterNd(flow.unittest.TestCase):
    @globaltest
    def test_scatter_nd(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_partial_sum=True, except_split=True):
                _test_scatter_nd(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
