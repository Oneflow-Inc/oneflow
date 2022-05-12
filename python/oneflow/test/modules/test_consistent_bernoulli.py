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
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_bernoulli(test_case, ndim, placement, sbp):
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).oneflow
    with flow.no_grad():
        x[:] = 1
    x = x.to_global(placement=placement, sbp=sbp)
    y = flow.bernoulli(x)
    test_case.assertTrue(np.allclose(y.numpy(), x.numpy()))


def _test_bernoulli_with_generator(test_case, ndim, placement, sbp):
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    generator = flow.Generator()
    generator.manual_seed(0)
    x = random_tensor(ndim, *dims).oneflow
    x = x.to_global(placement=placement, sbp=sbp)
    y0 = flow.bernoulli(x, generator=generator)
    generator.manual_seed(0)
    y1 = flow.bernoulli(x, generator=generator)
    test_case.assertTrue(np.allclose(y0.numpy(), y1.numpy()))


class TestBernoulli(flow.unittest.TestCase):
    @globaltest
    def test_bernoulli(test_case):
        for placement in all_placement():
            # bernoulli only has cpu kernel
            if placement.type != "cpu":
                continue
            ndim = random(1, 4).to(int).value()
            for sbp in all_sbp(placement, max_dim=min(2, ndim)):
                _test_bernoulli(test_case, ndim, placement, sbp)
                _test_bernoulli_with_generator(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
