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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(check_graph=False)
def _test_consistent_flow_tile_with_random_data(test_case, placement, sbp):
    x = random_pytorch_tensor(ndim=2, dim0=8, dim1=16).to_consistent(placement, sbp)
    reps = (
        random(1, 5).to(int) * 8,
        random(1, 5).to(int) * 8,
        random(1, 5).to(int) * 8,
    )
    z = torch.tile(x, reps)
    return z


@autotest(check_graph=False)
def _test_consistent_flow_tensor_tile_with_random_data(test_case, placement, sbp):
    x = random_pytorch_tensor(ndim=2, dim0=8, dim1=16).to_consistent(placement, sbp)
    reps = (
        random(1, 5).to(int) * 8,
        random(1, 5).to(int) * 8,
        random(1, 5).to(int) * 8,
    )
    y = x.tile(reps)
    return y


# ndsbp infer ger error when world_size != 1
@flow.unittest.skip_unless_1n1d()
class TestConsistentTile(flow.unittest.TestCase):
    @consistent
    def test_consistent_flow_tile_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_consistent_flow_tile_with_random_data(test_case, placement, sbp)

    @consistent
    def test_consistent_flow_tensor_tile_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_consistent_flow_tensor_tile_with_random_data(
                    test_case, placement, sbp
                )


if __name__ == "__main__":
    unittest.main()
