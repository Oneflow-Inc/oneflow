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


@autotest(n=1, check_graph=False)
def _test_prod_without_dim(test_case, placement, sbp, device, ndim):
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to(device=device)
    x = x.to_global(placement=placement, sbp=sbp)
    y = torch.prod(x)
    return y


class TestReduceProd(flow.unittest.TestCase):
    @globaltest
    def test_reduce_prod_without_dim(test_case):
        device = random_device()
        ndim = random(2, 5).to(int)
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_prod_without_dim(test_case, placement, sbp, device, ndim)


if __name__ == "__main__":
    unittest.main()
