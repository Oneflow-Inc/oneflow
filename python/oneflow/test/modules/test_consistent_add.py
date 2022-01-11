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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestAddModule(flow.unittest.TestCase):
    @consistent_autotest(check_graph=False)
    def test_0_size_add(test_case):
        try:
            placement = random_placement()
            sbp = random_sbp(placement, max_dim=2)
            x = random_pytorch_tensor(2, 0, 3)
            y = random_pytorch_tensor(2, 1, 3)
            x = x.to_consistent(placement=placement, sbp=sbp, grad_sbp=x.sbp)
            y = y.to_consistent(placement=placement, sbp=sbp, grad_sbp=y.sbp)
            out = x + y
            return out
        except Exception as e:
            print(
                "Failed to apply add operation on x and y with (placement: %s, sbp: %s)"
                % (placement.value(), sbp.value()),
            )
            assert "can't find available sbp signature." in str(e)

    # @autotest(auto_backward=False, check_graph=False)
    # def test_0dim_inplace_add(test_case):
    #     try:
    #         placement = random_placement()
    #         sbp = random_sbp(placement, max_dim=2)
    #         x = random_pytorch_tensor(2, 2, 3, requires_grad=False).to_consistent(placement=placement, sbp=sbp)
    #         y = random_pytorch_tensor(1, 10).to_consistent(placement=placement, sbp=random_sbp(placement, max_dim=1))
    #         x += y.mean()
    #         return x
    #     except Exception as e:
    #         assert 'can\'t find available sbp signature.' in str(e)

    # @autotest(check_graph=False)
    # def test_0dim_two_inplace_add(test_case):
    #     try:
    #         placement = random_placement()
    #         sbp = random_sbp(placement, max_dim=2)
    #         x = random_pytorch_tensor(2, 2, 3).to_consistent(placement=placement, sbp=sbp).mean()
    #         y = random_pytorch_tensor(2, 2, 3).to_consistent(placement=placement, sbp=sbp)
    #         x += y.mean()
    #         return x
    #     except Exception as e:
    #         assert 'can\'t find available sbp signature.' in str(e)

    @consistent_autotest(check_graph=False)
    def test_add_with_alpha(test_case):
        try:
            placement = random_placement()
            sbp = random_sbp(placement, max_dim=2)
            x1 = (
                random_pytorch_tensor(2, 2, 3)
                .to_consistent(placement=placement, sbp=sbp)
                .mean()
            )
            x2 = (
                random_pytorch_tensor(2, 2, 3)
                .to_consistent(placement=placement, sbp=sbp)
                .mean()
            )
            x3 = (
                random_pytorch_tensor(2, 2, 3)
                .to_consistent(placement=placement, sbp=sbp)
                .mean()
            )
            y = random_pytorch_tensor(2, 2, 3).to_consistent(
                placement=placement, sbp=sbp
            )
            s = random().to(float)
            alpha = random().to(float)
            z1 = torch.add(x1, y, alpha=alpha)
            z2 = torch.add(x2, s, alpha=alpha)
            z3 = torch.add(s, x3, alpha=alpha)
            return z1, z2, z3
        except Exception as e:
            print(
                "Failed to apply add operation on x and y with (placement: %s, sbp: %s)"
                % (placement.value(), sbp.value()),
            )
            # assert "can't find available sbp signature." in str(e)


if __name__ == "__main__":
    unittest.main()
