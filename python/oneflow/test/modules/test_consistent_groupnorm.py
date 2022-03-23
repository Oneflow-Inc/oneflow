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


@autotest(n=1, rtol=1e-03, atol=1e-03, check_graph=False)
def _test_group_norm(test_case, placement, sbp):
    dims = [random(1, 3).to(int) * 8 for _ in range(4)]
    channels = dims[1]
    m = torch.nn.GroupNorm(
        num_groups=oneof(1, 2, 4, 8),
        num_channels=channels,
        eps=random(0, 1).to(float),
        affine=random_bool(),
    )
    m.train(random_bool())
    m.to_global(placement=placement, sbp=random_sbp(placement, max_dim=0))
    x = random_tensor(4, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


class TestGroupNorm(flow.unittest.TestCase):
    @globaltest
    def test_groupnorm(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_group_norm(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
