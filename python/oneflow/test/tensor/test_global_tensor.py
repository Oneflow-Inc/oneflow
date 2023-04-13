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
import oneflow.unittest


from oneflow.test_utils.automated_test_util import *


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensor(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_creating_global_tensor(test_case):
        placement = flow.placement("cuda", [0])
        sbp = flow.sbp.broadcast

        # Shape -> GlobalTensor
        shape = (2, 3)
        x = flow.Tensor(*shape, placement=placement, sbp=sbp)
        test_case.assertTrue(x.is_global)
        test_case.assertTrue(x.size() == shape)

        shape = flow.Size((2, 3))
        x = flow.Tensor(shape, placement=placement, sbp=sbp)
        test_case.assertTrue(x.is_global)
        test_case.assertTrue(x.size() == shape)

        # LocalTensor -> GlobalTensor
        x = flow.Tensor(*shape, device="cpu")
        test_case.assertTrue(x.is_local)
        y = flow.Tensor(x, placement=placement, sbp=sbp)
        test_case.assertTrue(y.is_global)

        # GlobalTensor -> GlobalTensor
        z = flow.Tensor(y, placement=placement, sbp=sbp)
        test_case.assertTrue(z.is_global)

        # TODO: ndarray -> GlobalTensor

    @flow.unittest.skip_unless_1n1d()
    def test_construct_local_from_global_tensor(test_case):
        placement = flow.placement("cuda", [0])
        sbp = flow.sbp.broadcast
        shape = (2, 3)
        x = flow.Tensor(*shape, placement=placement, sbp=sbp)
        test_case.assertTrue(x.is_global)
        # GlobalTensor -> LocalTensor
        y = flow.Tensor(x, device="cpu")
        test_case.assertTrue(y.is_local)
        y = flow.Tensor(x, device="cuda")
        test_case.assertTrue(y.is_local)

    @flow.unittest.skip_unless_1n1d()
    def test_global_set_data(test_case):
        x_placement = flow.placement("cpu", [0])
        x_sbp = flow.sbp.broadcast
        x = flow.ones(2, 3, placement=x_placement, sbp=x_sbp)
        y_placement = flow.placement("cuda", [0])
        y_sbp = flow.sbp.split(0)
        y = flow.ones(4, 5, placement=y_placement, sbp=y_sbp)
        old_id = id(x)
        x.data = y
        test_case.assertEqual(old_id, id(x))
        test_case.assertTrue(x.shape == (4, 5))
        test_case.assertTrue(x.placement == y_placement)
        test_case.assertTrue(x.sbp[0] == y_sbp)

    @flow.unittest.skip_unless_1n1d()
    def test_global_tensor_autograd_related_methods(test_case):
        placement = flow.placement("cuda", [0])
        sbp = flow.sbp.split(0)
        shape = (2, 3, 4, 5)
        l_x = flow.Tensor(*shape)
        test_case.assertFalse(l_x.requires_grad)
        test_case.assertTrue(l_x.is_leaf)

        l_y = flow.Tensor(*shape)
        l_y.requires_grad = True
        test_case.assertTrue(l_y.requires_grad)
        test_case.assertTrue(l_y.is_leaf)

        x = l_x.to_global(placement=placement, sbp=sbp)
        test_case.assertTrue(x.is_leaf)
        y = l_y.to_global(placement=placement, sbp=sbp)
        test_case.assertFalse(y.is_leaf)

        z = x + y
        test_case.assertTrue(z.requires_grad)
        test_case.assertFalse(z.is_leaf)

        with flow.no_grad():
            m = x + y

        test_case.assertTrue(m.is_leaf)
        test_case.assertFalse(m.requires_grad)

        l_v = flow.Tensor(*shape)
        l_v.requires_grad = True
        v = l_v.to_global(placement=placement, sbp=sbp)

        z.retain_grad()
        w = v + z

        l_grad = flow.ones(*shape)
        grad = l_grad.to_global(placement=placement, sbp=sbp)
        w.backward(gradient=grad)

        test_case.assertTrue(
            np.allclose(l_v.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(
            np.allclose(l_y.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(
            np.allclose(
                z.grad.to_global(sbp=flow.sbp.broadcast).to_local().numpy(),
                np.ones(shape),
                atol=1e-4,
                rtol=1e-4,
            )
        )
        test_case.assertIsNone(l_x.grad)

    @flow.unittest.skip_unless_1n1d()
    def test_global_tensor_unsupported_property(test_case):

        shape = (2, 3)
        placement = flow.placement("cuda", [0])
        sbp = flow.sbp.split(0)
        a = flow.Tensor(*shape)
        b = a.to_global(placement=placement, sbp=sbp)
        test_case.assertTrue(b.is_global)

        with test_case.assertRaises(RuntimeError):
            b.device()

        with test_case.assertRaises(RuntimeError):
            b._tensor_buffer_shapes_and_dtypes

    @flow.unittest.skip_unless_1n4d()
    def test_global_tensor_2d_sbp_init(test_case):
        V = 10
        H = 4
        S = 6

        P = flow.placement("cuda", [[0, 1], [2, 3]])

        wte = flow.nn.Parameter(
            flow.empty(
                (V, H),
                dtype=flow.float32,
                placement=P,
                sbp=[flow.sbp.broadcast, flow.sbp.split(0)],
            )
        )

        wpe = flow.nn.Parameter(
            flow.empty(
                (S, H),
                dtype=flow.float32,
                placement=P,
                sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            )
        )

        flow.nn.init.normal_(wte, std=0.02)
        flow.nn.init.normal_(wpe, std=0.02)

    @flow.unittest.skip_unless_1n2d()
    def test_copy(test_case):
        x = flow.zeros(2, 3)
        y = flow.ones(2, 3)
        x.copy_(y)
        test_case.assertTrue(np.array_equal(x.numpy(), y.numpy()))

        x = flow.zeros(
            4, 6, placement=flow.placement("cuda", [0, 1]), sbp=flow.sbp.broadcast
        )
        y = flow.ones(
            4, 6, placement=flow.placement("cpu", [0]), sbp=flow.sbp.broadcast
        )
        x.copy_(y)
        test_case.assertTrue(np.array_equal(x.numpy(), y.numpy()))

        x = flow.zeros(
            4, 6, placement=flow.placement("cuda", [0, 1]), sbp=flow.sbp.broadcast
        )
        y = flow.ones(
            4, 6, placement=flow.placement("cuda", [0]), sbp=flow.sbp.broadcast
        )
        x.copy_(y)
        test_case.assertTrue(np.array_equal(x.numpy(), y.numpy()))

        x = flow.zeros(
            4, 6, placement=flow.placement("cuda", [0, 1]), sbp=flow.sbp.split(0)
        )
        y = flow.ones(
            4, 6, placement=flow.placement("cuda", [0, 1]), sbp=flow.sbp.broadcast
        )
        x.copy_(y)
        test_case.assertTrue(np.array_equal(x.numpy(), y.numpy()))

        x = flow.zeros(
            4, 6, placement=flow.placement("cuda", [0, 1]), sbp=flow.sbp.broadcast
        )
        y = flow.ones(
            4, 6, placement=flow.placement("cuda", [0, 1]), sbp=flow.sbp.broadcast
        )
        x.copy_(y)
        test_case.assertTrue(np.array_equal(x.numpy(), y.numpy()))


if __name__ == "__main__":
    unittest.main()
