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


def _test_broadcast_like(test_case, placement, sbp):
    like_shape = [8] * 4
    like = random_tensor(4, *like_shape).to_global(
        placement, random_sbp(placement, max_dim=4)
    )
    x = random_tensor(2, *(8, 8)).to_global(placement, sbp)
    # oneflow
    of_y = flow.broadcast_like(x.oneflow, like.oneflow)
    # pytorch
    torch_y = x.pytorch.broadcast_to(like_shape)

    test_case.assertTrue(np.allclose(of_y.numpy(), torch_y.detach().cpu().numpy()))


def _test_broadcast_like_expand_dims(test_case, placement, sbp):
    like_shape = [8] * 4
    like = random_tensor(4, *like_shape).to_global(
        placement, random_sbp(placement, max_dim=4)
    )
    x = random_tensor(2, *(8, 8)).to_global(placement, sbp)
    # oneflow
    of_y = flow.broadcast_like(x.oneflow, like.oneflow, [1, 3])
    # pytorch
    torch_y = x.pytorch.view(8, 1, 8, 1).broadcast_to(like_shape)

    test_case.assertTrue(np.allclose(of_y.numpy(), torch_y.detach().cpu().numpy()))


class TestGlobalBroadcaseLike(flow.unittest.TestCase):
    @globaltest
    def test_broadcase_like(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_broadcast_like(test_case, placement, sbp)
                _test_broadcast_like_expand_dims(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
