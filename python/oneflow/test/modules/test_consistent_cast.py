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
import os

import oneflow.unittest
from test_util import GenArgList


@flow.unittest.skip_unless_1n4d()
class TestConsistentCastModule_1n4d(flow.unittest.TestCase):
    def test_to_consistent_broadcast_shape_dtype(test_case):
        if int(os.getenv("RANK")) < 2:
            x = flow.ones((16, 16), dtype=flow.int32)
        else:
            x = flow.zeros((1,), dtype=flow.float)
        placement = flow.placement("cpu", {0: range(2)})
        sbp = (flow.sbp.split(0),)
        y = x.to_consistent(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (32, 16))
        test_case.assertEqual(y.dtype, flow.int32)


@flow.unittest.skip_unless_1n2d()
class TestConsistentCastModule_1n2d(flow.unittest.TestCase):
    def test_to_consistent_broadcast_shape_dtype(test_case):
        if os.getenv("RANK") == "0":
            x = flow.ones((16, 16), dtype=flow.int32)
        else:
            x = flow.zeros((1,), dtype=flow.float)
        placement = flow.placement("cpu", {0: [0]})
        sbp = (flow.sbp.broadcast,)
        y = x.to_consistent(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (16, 16))
        test_case.assertEqual(y.dtype, flow.int32)

    def test_local_to_consistent_broadcast_data(test_case):
        if int(os.getenv("RANK")) == 0:
            x = flow.ones((16, 16), dtype=flow.int32)
        else:
            x = flow.zeros((16, 16), dtype=flow.int32)
        placement = flow.placement("cpu", {0: range(2)})
        sbp = (flow.sbp.broadcast,)
        y = x.to_consistent(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (16, 16))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((16, 16), dtype=np.int32))
        )

    def test_cuda_consistent_to_consistent_s2b(test_case):
        x = flow.ones((16, 16), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", {0: range(2)})
        y = x.to_consistent(placement=placement, sbp=flow.sbp.split(0))
        sbp = (flow.sbp.broadcast,)
        y = y.to_consistent(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (32, 16))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((32, 16), dtype=np.int32))
        )

    def test_cuda_consistent_to_consistent_s2p(test_case):
        x = flow.ones((16, 16), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", {0: range(2)})
        y = x.to_consistent(placement=placement, sbp=flow.sbp.split(0))
        sbp = (flow.sbp.partial_sum,)
        y = y.to_consistent(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (32, 16))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        if int(os.getenv("RANK")) == 0:
            test_case.assertTrue(
                np.array_equal(z.numpy(), np.ones((32, 16), dtype=np.int32))
            )
        else:
            test_case.assertTrue(
                np.array_equal(z.numpy(), np.zeros((32, 16), dtype=np.int32))
            )

    def test_cuda_consistent_to_consistent_b2p(test_case):
        x = flow.ones((16, 16), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", {0: range(2)})
        y = x.to_consistent(placement=placement, sbp=flow.sbp.broadcast)
        sbp = (flow.sbp.partial_sum,)
        y = y.to_consistent(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (16, 16))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        if int(os.getenv("RANK")) == 0:
            test_case.assertTrue(
                np.array_equal(z.numpy(), np.ones((16, 16), dtype=np.int32))
            )
        else:
            test_case.assertTrue(
                np.array_equal(z.numpy(), np.zeros((16, 16), dtype=np.int32))
            )

    def test_cuda_consistent_to_consistent_b2s(test_case):
        x = flow.ones((16, 16), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", {0: range(2)})
        y = x.to_consistent(placement=placement, sbp=flow.sbp.broadcast)
        sbp = (flow.sbp.split(0),)
        y = y.to_consistent(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (16, 16))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((8, 16), dtype=np.int32))
        )

    def test_cuda_consistent_to_consistent_p2s(test_case):
        x = flow.ones((16, 16), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", {0: range(2)})
        y = x.to_consistent(placement=placement, sbp=flow.sbp.partial_sum)
        sbp = (flow.sbp.split(0),)
        y = y.to_consistent(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (16, 16))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((8, 16), dtype=np.int32) * 2)
        )

    def test_cuda_consistent_to_consistent_p2b(test_case):
        x = flow.ones((16, 16), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", {0: range(2)})
        y = x.to_consistent(placement=placement, sbp=flow.sbp.partial_sum)
        sbp = (flow.sbp.broadcast,)
        y = y.to_consistent(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (16, 16))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((16, 16), dtype=np.int32) * 2)
        )


@flow.unittest.skip_unless_1n1d()
class TestConsistentCastModule_1n1d(flow.unittest.TestCase):
    def test_to_consistent(test_case):
        x = flow.ones((16, 16))
        placement = flow.placement("cpu", {0: [0]})
        sbp = (flow.sbp.broadcast,)
        y = x.to_consistent(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (16, 16))


if __name__ == "__main__":
    unittest.main()
