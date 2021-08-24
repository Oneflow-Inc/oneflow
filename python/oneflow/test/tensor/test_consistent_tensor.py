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
from automated_test_util import *

import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensor(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_creating_consistent_tensor(test_case):
        placement = flow.placement("cuda", {0: 0})
        sbp = flow.sbp.broadcast
        shape = (2, 3)

        # Shape -> ConsistentTensor
        x = flow.Tensor(*shape, placement=placement, sbp=sbp)
        test_case.assertTrue(x.is_consistent)

        # LocalTensor -> ConsistentTensor
        x = flow.Tensor(*shape, device="cpu")
        test_case.assertTrue(x.is_local)
        y = flow.Tensor(x, placement=placement, sbp=sbp)
        test_case.assertTrue(y.is_consistent)

        # ConsistentTensor -> ConsistentTensor
        z = flow.Tensor(y, placement=placement, sbp=sbp)
        test_case.assertTrue(z.is_consistent)

        # TODO: ndarray -> ConsistentTensor

    @flow.unittest.skip_unless_1n1d()
    def test_construct_local_from_consistent_tensor(test_case):
        placement = flow.placement("cuda", {0: 0})
        sbp = flow.sbp.broadcast
        shape = (2, 3)
        x = flow.Tensor(*shape, placement=placement, sbp=sbp)
        test_case.assertTrue(x.is_consistent)
        # ConsistentTensor -> LocalTensor
        y = flow.Tensor(x)
        test_case.assertTrue(y.is_local)
        y = flow.Tensor(x, device="cuda")
        test_case.assertTrue(y.is_local)

    @flow.unittest.skip_unless_1n1d()
    def test_consistent_tensor_autograd_related_methods(test_case):
        placement = flow.placement("cuda", {0: 0})
        sbp = flow.sbp.split(0)
        shape = (2, 3, 4, 5)
        l_x = flow.Tensor(*shape)
        test_case.assertFalse(l_x.requires_grad)
        test_case.assertTrue(l_x.is_leaf)

        l_y = flow.Tensor(*shape, requires_grad=True)
        test_case.assertTrue(l_y.requires_grad)
        test_case.assertTrue(l_y.is_leaf)

        x = l_x.to_consistent(placement=placement, sbp=sbp)
        test_case.assertTrue(x.is_leaf)
        y = l_y.to_consistent(placement=placement, sbp=sbp)
        test_case.assertFalse(y.is_leaf)

        z = x + y
        test_case.assertTrue(z.requires_grad)
        test_case.assertFalse(z.is_leaf)

        with flow.no_grad():
            m = x + y

        test_case.assertTrue(m.is_leaf)
        test_case.assertFalse(m.requires_grad)

        l_v = flow.Tensor(*shape, requires_grad=True)
        v = l_v.to_consistent(placement=placement, sbp=sbp)

        z.retain_grad()
        w = v + z

        l_grad = flow.ones(*shape)
        grad = l_grad.to_consistent(placement=placement, sbp=sbp)
        w.backward(gradient=grad)

        test_case.assertTrue(
            np.allclose(l_v.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(
            np.allclose(l_y.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(
            np.allclose(
                z.grad.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy(),
                np.ones(shape),
                atol=1e-4,
                rtol=1e-4,
            )
        )
        test_case.assertIsNone(l_x.grad)

    @flow.unittest.skip_unless_1n1d()
    def test_consistent_tensor_unsupported_property(test_case):

        shape = (2, 3)
        placement = flow.placement("cuda", {0: 0})
        sbp = flow.sbp.split(0)
        a = flow.Tensor(*shape)
        b = a.to_consistent(placement=placement, sbp=sbp)
        test_case.assertTrue(b.is_consistent)

        with test_case.assertRaises(
            oneflow._oneflow_internal.exception.RuntimeException
        ):
            b.device()

        with test_case.assertRaises(
            oneflow._oneflow_internal.exception.RuntimeException
        ):
            b._tensor_buffer_shapes_and_dtypes


if __name__ == "__main__":
    unittest.main()
