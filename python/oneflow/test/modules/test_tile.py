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


@flow.unittest.skip_unless_1n1d()
class TestTile(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_tile_with_random_data(test_case):
        x = random_tensor(ndim=2, dim0=1, dim1=2)
        reps = (random(1, 5).to(int), random(1, 5).to(int), random(1, 5).to(int))
        z = torch.tile(x, reps)
        return z

    @autotest(check_graph=True)
    def test_flow_tensor_tile_with_random_data(test_case):
        x = random_tensor(ndim=2, dim0=1, dim1=2)
        reps = (random(1, 5).to(int), random(1, 5).to(int), random(1, 5).to(int))
        y = x.tile(reps)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_tile_bool_with_random_data(test_case):
        x = random_tensor(ndim=2, dim0=1, dim1=2).to(torch.bool)
        reps = (random(1, 5).to(int), random(1, 5).to(int), random(1, 5).to(int))
        z = torch.tile(x, reps)
        return z

    @autotest(check_graph=True)
    def test_flow_tile_with_0dim_data(test_case):
        x = random_tensor(ndim=0)
        reps = (random(1, 5).to(int), random(1, 5).to(int), random(1, 5).to(int))
        z = torch.tile(x, reps)
        return z


if __name__ == "__main__":
    unittest.main()
