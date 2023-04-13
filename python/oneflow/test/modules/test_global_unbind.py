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


# TODO: the test is dependent on global select op(global tensor->stride())
@unittest.skip("global select op is not currently supported")
@autotest(n=1, check_graph=True)
def _test_unbind(test_case, placement, sbp):
    dim_size = random(1, 3).to(int).value() * 8
    rand_dim = random(0, 3).to(int).value()
    x = random_tensor(ndim=3, dim0=dim_size, dim1=dim_size, dim2=dim_size).to_global(
        placement, sbp
    )
    return torch.unbind(x, dim=rand_dim)


class TestUnbind(flow.unittest.TestCase):
    @globaltest
    def test_unbind(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_unbind(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
