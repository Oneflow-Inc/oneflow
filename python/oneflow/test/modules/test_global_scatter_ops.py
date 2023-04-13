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


@autotest(n=10, auto_backward=True, check_graph=True)
def _test_scatter_random_data(test_case, placement):
    input = random_tensor(ndim=2, dim0=2, dim1=2).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    src = random_tensor(ndim=2, dim0=2, dim1=2).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    index = (
        torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.int64)
        .to_global(flow.placement.all("cpu"), [flow.sbp.broadcast,])
        .to_global(placement, sbp=random_sbp(placement, max_dim=2),)
    )
    dim = random(0, 2).to(int).value()
    return torch.scatter(input, dim, index, src)


@autotest(n=10, auto_backward=True, check_graph=True)
def _test_scatter_scalar_random_data(test_case, placement):
    input = random_tensor(ndim=2, dim0=2, dim1=2).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    index = (
        torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.int64)
        .to_global(flow.placement.all("cpu"), [flow.sbp.broadcast,])
        .to_global(placement, sbp=random_sbp(placement, max_dim=2),)
    )
    dim = random(0, 2).to(int).value()
    return torch.scatter(input, dim, index, 3.14)


@autotest(n=10, auto_backward=True, check_graph=True)
def _test_scatter_add_random_data(test_case, placement):
    input = random_tensor(ndim=2, dim0=2, dim1=2).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    src = random_tensor(ndim=2, dim0=2, dim1=2).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    index = (
        torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.int64)
        .to_global(flow.placement.all("cpu"), [flow.sbp.broadcast,])
        .to_global(placement, sbp=random_sbp(placement, max_dim=2),)
    )
    dim = random(0, 2).to(int).value()
    return torch.scatter_add(input, dim, index, src)


@flow.unittest.skip_unless_1n2d()
class TestScatterOps(flow.unittest.TestCase):
    @globaltest
    def test_scatter_ops(test_case):
        for placement in all_placement():
            _test_scatter_random_data(test_case, placement)
            _test_scatter_scalar_random_data(test_case, placement)
            _test_scatter_add_random_data(test_case, placement)


if __name__ == "__main__":
    unittest.main()
