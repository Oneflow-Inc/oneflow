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


def _test_randint(test_case, placement, sbp, device, shape, low, high):
    y1 = flow.randint(low, high, shape, device=flow.device(device))
    y2 = flow.randint(low, high, shape, device=flow.device(device))
    y1 = y1.to_consistent(placement=placement, sbp=sbp)
    y2 = y2.to_consistent(placement=placement, sbp=sbp)
    if flow.env.get_rank() == 0:
        test_case.assertFalse(np.allclose(y1.to_local().numpy(), y2.to_local().numpy(), atol=1e-4, rtol=1e-4))
        test_case.assertTrue(shape == y1.to_local().shape)

def _test_different_dtype(test_case, placement, sbp, device, shape, low, high):
    for dtype in [
        # flow.uint8,
        flow.int8,
        flow.int32,
        flow.int64,
        flow.float32,
        flow.float64,
    ]:
        y = flow.randint(low, high, shape, dtype=dtype, device=flow.device(device))
        y = y.to_consistent(placement=placement, sbp=sbp)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(y.to_local().dtype == dtype)
            test_case.assertTrue(y.to_local().shape == shape)

def _test_with_generator(test_case, placement, sbp, device, shape, low, high):
    gen = flow.Generator()
    gen.manual_seed(0)
    y1 = flow.randint(
        low, high, shape, dtype=flow.float32, device=flow.device(device), generator=gen
    )
    gen.manual_seed(0)
    y2 = flow.randint(
        low, high, shape, dtype=flow.float32, device=flow.device(device), generator=gen
    )
    y1 = y1.to_consistent(placement=placement, sbp=sbp)
    y2 = y2.to_consistent(placement=placement, sbp=sbp)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(np.allclose(y1.to_local().numpy(), y2.to_local().numpy(), atol=1e-4, rtol=1e-4))


def _test_0d_randint(test_case, placement, sbp, device, shape, low, high):
    y1 = flow.randint(low, high, shape, device=flow.device(device))
    y2 = flow.randint(low, high, shape, device=flow.device(device))
    y1 = y1.to_consistent(placement=placement, sbp=sbp)
    y2 = y2.to_consistent(placement=placement, sbp=sbp)
    
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.allclose(y1.to_local().numpy(), y2.to_local().numpy(), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(shape == y1.to_local().shape)

def _test_high(test_case, placement, sbp, device, shape, low, high):
    y1 = flow._C.randint(high, shape, device=flow.device(device))
    y2 = flow._C.randint(high, shape, device=flow.device(device))
    y1 = y1.to_consistent(placement=placement, sbp=sbp)
    y2 = y2.to_consistent(placement=placement, sbp=sbp)
    if flow.env.get_rank() == 0:
        test_case.assertFalse(np.allclose(y1.to_local().numpy(), y2.to_local().numpy(), atol=1e-4, rtol=1e-4))
        test_case.assertTrue(shape == y1.to_local().shape)

class TestRandint(flow.unittest.TestCase):
    @consistent
    def test_consistent_different_types(test_case):
        for dtype in [
            flow.uint8,
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float32,
            flow.float64,
        ]:
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=2):
                    x = flow.randint(0, 16, (10, 1), dtype=dtype)
                    x = x.to_consistent(placement=placement, sbp=sbp)
                    if flow.env.get_rank() == 0:
                        test_case.assertEqual(x.dtype, dtype)
                        test_case.assertEqual(x.sbp, sbp)
                        test_case.assertEqual(x.placement, placement)

    @consistent
    def test_randint(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_randint,
            _test_different_dtype,
            _test_with_generator,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["low"] = [i for i in range(10)]
        arg_dict["high"] = [10 + np.random.randint(10, 20) for i in range(10)]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=2):
                    arg[0](test_case, placement, sbp, *arg[1:])
    
    @consistent
    def test_0d_randint(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_0d_randint]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 0, 4), (2, 0, 2)]
        arg_dict["low"] = [i for i in range(10)]
        arg_dict["high"] = [10 + np.random.randint(1, 20) for i in range(10)]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=2):
                    arg[0](test_case, placement, sbp, *arg[1:])

    @consistent
    def test_high_randint(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_high]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 3, 4), (2, 5, 2)]
        arg_dict["low"] = [i for i in range(10)]
        arg_dict["high"] = [10 + np.random.randint(10, 20) for i in range(10)]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=2):
                    arg[0](test_case, placement, sbp, *arg[1:])

if __name__ == "__main__":
    unittest.main()

#uint8类型有问题