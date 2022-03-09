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
from test_util import GenArgList
from oneflow.test_utils.automated_test_util import *


@autotest(check_graph=False, auto_backward=False)
def _test_clip_grad_norm_impl(test_case, shape, max_norm, norm_type, placement, sbp):
    input = random_pytorch_tensor(
        len(shape), *shape
    ).to_consistent(
        placement, sbp
    )
    m = torch.nn.ReLU()
    m(input).sum().backward()
    return torch.nn.utils.clip_grad_norm_(input, max_norm, norm_type)


@autotest(check_graph=False, auto_backward=False)
def _test_clip_grad_value_impl(test_case, shape, clip_value, placement, sbp):
    input = random_pytorch_tensor(
        len(shape), *shape
    ).to_consistent(
        placement, sbp
    )
    m = torch.nn.ReLU()
    m(input).sum().backward()
    return torch.nn.utils.clip_grad_value_(input, clip_value)


class TestClipGrad(flow.unittest.TestCase):
    @consistent
    def test_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["max_norm"] = [0, 0.5, 1.0]
        arg_dict["norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=4):
                    _test_clip_grad_norm_impl(test_case, *arg, placement, sbp)

    def test_clip_value(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["clip_value"] = [0, 0.5, 1.0]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=4):
                    _test_clip_grad_value_impl(test_case, *arg, placement, sbp)


if __name__ == "__main__":
    unittest.main()
