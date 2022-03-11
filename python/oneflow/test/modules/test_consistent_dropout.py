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

@autotest(n=1, auto_backward=True, check_graph=False)
def _autotest_dropout_p0(test_case, placement, sbp, p):
    ndim = 1
    dims = [random(1, 5) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims)
    y = x.to_global(placement=placement, sbp=sbp)
    m = torch.nn.Dropout(p=p, inplace=random_bool())
    return m(x)

@autotest(n=1, auto_backward=True, check_graph=False)
def _autotest_dropout_eval(test_case, placement, sbp, p):
    ndim = 1
    dims = [random(1, 5) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims)
    y = x.to_global(placement=placement, sbp=sbp)
    m = torch.nn.Dropout(p=p, inplace=random_bool())
    m.eval()
    return m(x)

class TestDropoutConsistent(flow.unittest.TestCase):
    @globaltest
    def autotest_dropout_p01(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _autotest_dropout_p01(test_case, placement, sbp, p=0.0)
                _autotest_dropout_p01(test_case, placement, sbp, p=1.0)

    @globaltest
    def autotest_dropout_eval(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _autotest_dropout_eval(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
