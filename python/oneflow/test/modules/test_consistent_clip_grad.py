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

import oneflow as flow
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=False, auto_backward=False)
def _test_clip_grad_norm_impl(test_case, ndim, placement, sbp):
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    input = random_tensor(ndim, *dims).clone().to_global(placement=placement, sbp=sbp)
    m = torch.nn.ReLU()
    m(input).sum().backward()

    max_norm = oneof(0, 0.5, 1.0)
    norm_type = oneof("inf", "-inf", 0.0, 1.0, 2.0, 3.5)
    return torch.nn.utils.clip_grad_norm_(input, max_norm, norm_type)


@autotest(n=1, check_graph=False, auto_backward=False)
def _test_clip_grad_value_impl(test_case, ndim, placement, sbp):
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    input = random_tensor(ndim, *dims).clone().to_global(placement=placement, sbp=sbp)
    m = torch.nn.ReLU()
    m(input).sum().backward()

    clip_value = oneof(0, 0.5, 1.0)
    return torch.nn.utils.clip_grad_value_(input, clip_value)


class TestClipGrad(flow.unittest.TestCase):
    @globaltest
    def test_clip_grad(test_case):
        for placement in all_placement():
            ndim = random(1, 4).to(int).value()
            for sbp in all_sbp(placement, max_dim=min(ndim, 2)):
                _test_clip_grad_norm_impl(test_case, ndim, placement, sbp)

    @globaltest
    def test_clip_value(test_case):
        for placement in all_placement():
            ndim = random(1, 4).to(int).value()
            for sbp in all_sbp(placement, max_dim=min(ndim, 2)):
                _test_clip_grad_value_impl(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
