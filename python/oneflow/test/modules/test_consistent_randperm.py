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

from random import shuffle
import numpy as np
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *

def _test_randperm_with_generator(test_case, placement, sbp, N, device, dtype):
    generator = flow.Generator()
    generator.manual_seed(0)
    y_1 = flow.randperm(N, device=device, dtype=dtype, generator=generator)
    generator.manual_seed(0)
    y_2 = flow.randperm(N, device=device, dtype=dtype, generator=generator)
    y_1 = y_1.to_consistent(placement=placement, sbp=sbp)
    y_2 = y_2.to_consistent(placement=placement, sbp=sbp)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(np.allclose(y_1.to_local().numpy(), y_2.to_local().numpy()))
        test_case.assertTrue(y_1.to_local().dtype == dtype and y_2.to_local().dtype == dtype)


def _test_randperm_backward(test_case, placement, sbp, N, device, dtype):
    dtype = flow.float32  # fix dtype here as reduce_sum doesn't support all dtypes yet
    x = flow.randperm(N, device=device, dtype=dtype)
    x.requires_grad = True
    x = x.to_consistent(placement=placement, sbp=sbp)
    y = x.sum()
    y.backward()
    if flow.env.get_rank() == 0:
        test_case.assertTrue(np.allclose(x.grad.to_local().numpy(), np.ones(N), 1e-05, 1e-05))


def _test_randperm_randomness(test_case, placement, sbp, N, device, dtype):
    x1 = flow.randperm(N, device=device)
    x2 = flow.randperm(N, device=device)
    x1 = x1.to_consistent(placement=placement, sbp=sbp)
    x2 = x2.to_consistent(placement=placement, sbp=sbp)
    if flow.env.get_rank() == 0:
        test_case.assertFalse(np.all(x1.to_local().numpy() == x2.to_local().numpy()))


class Testrandperm(flow.unittest.TestCase):
    @consistent
    def test_consistent_different_types(test_case):
        for dtype in [
            #flow.uint8,
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float32,
            flow.float64,
        ]:
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=1):
                    x = flow.randperm(10,dtype=dtype)
                    x = x.to_consistent(placement=placement,sbp=sbp)
                    if flow.env.get_rank() == 0:
                        test_case.assertEqual(x.dtype, dtype)
                        test_case.assertEqual(x.sbp, sbp)
                        test_case.assertEqual(x.placement, placement)
    
    @consistent
    def test_randperm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_functions"] = [
            _test_randperm_with_generator,
            _test_randperm_randomness,
        ]
        arg_dict["N"] = [i for i in range(10, 100, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [
            #flow.uint8,
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float32,
            flow.float64,
        ]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=1):
                    arg[0](test_case, placement, sbp, *arg[1:])
    
    @consistent
    def test_randperm_backward(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_functions"] = [
            _test_randperm_backward,
        ]
        arg_dict["N"] = [i for i in range(10, 100, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [flow.float32, flow.float64]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=1):
                    arg[0](test_case, placement, sbp, *arg[1:])

    @consistent
    def test_auto_1(test_case):
        device = random_device()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                y = torch.randperm(1, device=device)
                y = y.to_consistent(placement=placement, sbp=sbp)
                return y


if __name__ == "__main__":
    unittest.main()

#flow.uint8类型不支持