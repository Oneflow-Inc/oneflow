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


class TestModule(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_get_sbp_with_invalid_axis(test_case):
        with test_case.assertRaises(RuntimeError) as ctx:
            sbp = flow.sbp.split(-1)
        test_case.assertTrue(
            "Split axis must not be negative, but got -1!" in str(ctx.exception)
        )

        with test_case.assertRaises(RuntimeError) as ctx:
            sbp = flow.sbp.split(7)
        test_case.assertTrue(
            "Expected split axis to be less than the supported maximum axis (6), but got 7!"
            in str(ctx.exception)
        )

    @flow.unittest.skip_unless_1n1d()
    def test_local_to_global_with_invalid_split_axis(test_case):
        x = flow.tensor([1, 2, 3, 4])
        with test_case.assertRaises(RuntimeError) as ctx:
            y = x.to_global(placement=flow.placement.all("cpu"), sbp=flow.sbp.split(1))
        test_case.assertTrue(
            "Split axis out of range (expected to be in range of [0, 1), but got 1!"
            in str(ctx.exception)
        )

    @flow.unittest.skip_unless_1n1d()
    def test_global_to_global_with_invalid_split_axis(test_case):
        x = flow.tensor(
            [1, 2, 3, 4], placement=flow.placement.all("cpu"), sbp=flow.sbp.broadcast,
        )
        with test_case.assertRaises(RuntimeError) as ctx:
            y = x.to_global(sbp=flow.sbp.split(1))
        test_case.assertTrue(
            "Split axis out of range (expected to be in range of [0, 1), but got 1!"
            in str(ctx.exception)
        )

    @flow.unittest.skip_unless_1n1d()
    def test_call_to_local_for_local_tensor(test_case):
        x = flow.tensor([1, 2, 3, 4])
        with test_case.assertRaises(RuntimeError) as ctx:
            y = x.to_local()
        test_case.assertTrue(
            "Expected global tensor for to_local but got local tensor!"
            in str(ctx.exception)
        )

    @flow.unittest.skip_unless_1n2d()
    def test_local_to_global_with_invalid_size(test_case):
        if flow.env.get_rank() == 0:
            x = flow.Tensor(2, 4)  # size(2, 4)
        else:
            x = flow.Tensor(4, 4)  # size(4, 4)
        with test_case.assertRaises(RuntimeError) as ctx:
            y = x.to_global(placement=flow.placement.all("cpu"), sbp=flow.sbp.split(0))
        test_case.assertTrue(
            "Sizes of tensors in dimension 0 must be same or match balanced split distribution. "
            "See https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/common/balanced_splitter.h "
            "for details of balanced split" in str(ctx.exception)
        )

        with test_case.assertRaises(RuntimeError) as ctx:
            y = x.to_global(placement=flow.placement.all("cpu"), sbp=flow.sbp.split(1))
        test_case.assertTrue(
            "Sizes of tensors must match except in dimension 1. Expected size 2 but got size 4 for tensor on rank 1!"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
