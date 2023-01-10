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


@autotest(n=1, check_graph=True)
def _test_reshape_impl(test_case, pair, placement, sbp):
    shape, to_shape = pair
    x = random_tensor(len(shape), *shape)
    y = x.to_global(placement=placement, sbp=sbp)
    z = y.reshape(to_shape)
    return z


def _test_reshape_like_impl(test_case, pair, placement, in_sbp, like_sbp):
    shape, to_shape = pair

    nd_arr = np.random.rand(*shape)
    np_out = nd_arr.reshape(to_shape)

    x = flow.tensor(nd_arr)
    like = flow.empty(to_shape)
    y = x.to_global(flow.placement.all("cpu"), flow.sbp.broadcast).to_global(
        placement=placement, sbp=in_sbp
    )
    like = like.to_global(flow.placement.all("cpu"), flow.sbp.broadcast).to_global(
        placement=placement, sbp=like_sbp
    )
    z = flow._C.reshape_like(y, like)
    local_z = z.to_global(
        placement, sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))]
    ).to_local()
    if flow.env.get_rank() == 0:
        test_case.assertTrue(np.array_equal(np_out, local_z.numpy()))


class TestReshapeGlobal(flow.unittest.TestCase):
    @globaltest
    def test_reshape(test_case):
        shape_pairs = [
            ((8, 16), (8 * 16,)),
            ((8, 16), (8 * 4, 4)),
            ((8, 16, 24), (64, 6, 8)),
            ((8, 16), (64, 1, -1)),
            ((8, 16), (-1,)),
        ]
        for pair in shape_pairs:
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=len(pair[0])):
                    _test_reshape_impl(test_case, pair, placement, sbp)

    @globaltest
    def test_reshape_like(test_case):
        shape_pairs = [
            ((8, 16), (8 * 16,)),
            ((8, 16), (8 * 2, 8)),
            ((8, 16, 24), (64, 48)),
        ]
        for pair in shape_pairs:
            for placement in all_placement():
                for in_sbp in all_sbp(placement, max_dim=len(pair[0])):
                    for like_sbp in all_sbp(placement, max_dim=len(pair[1])):
                        _test_reshape_like_impl(
                            test_case, pair, placement, in_sbp, like_sbp
                        )


if __name__ == "__main__":
    unittest.main()
