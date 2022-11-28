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
import os
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class Test2DeviceGlobalTensorTo(flow.unittest.TestCase):
    def test_asymmetric_global_tensor_clone(test_case):
        placement = flow.placement("cuda", range(1))
        x = flow.ones((4,), placement=placement, sbp=flow.sbp.broadcast)
        cloned = x.detach().clone()
        test_case.assertEqual(x.placement, cloned.placement)
        test_case.assertEqual(x.sbp, cloned.sbp)
        if flow.env.get_rank() == 0:
            cloned_local = cloned.to_local()
            cloned_local[0] = 0
            test_case.assertEqual(cloned_local[0].numpy().item(), 0)
            test_case.assertEqual(x.to_local()[0].numpy().item(), 1)

    def test_global_tensor_clone(test_case):
        placement = flow.placement("cuda", range(2))
        x = flow.ones((4,), placement=placement, sbp=flow.sbp.broadcast)
        cloned = x.detach().clone()
        test_case.assertEqual(x.placement, cloned.placement)
        test_case.assertEqual(x.sbp, cloned.sbp)
        cloned_local = cloned.to_local()
        cloned_local[0] = 0
        test_case.assertEqual(cloned_local[0].numpy().item(), 0)
        test_case.assertEqual(x.to_local()[0].numpy().item(), 1)

    def test_global_tensor_to(test_case):
        placement = flow.placement("cuda", range(2))
        x = flow.ones((4,), placement=placement, sbp=flow.sbp.broadcast)
        cloned = x.to(copy=True)
        test_case.assertEqual(x.placement, cloned.placement)
        test_case.assertEqual(x.sbp, cloned.sbp)
        cloned_local = cloned.to_local()
        cloned_local[0] = 0
        test_case.assertEqual(cloned_local[0].numpy().item(), 0)
        test_case.assertEqual(x.to_local()[0].numpy().item(), 1)

    def test_tensor_to_h2d1(test_case):
        input = flow.tensor(np.random.randn(2, 3, 4, 5), dtype=flow.int64)
        output = input.to(device=flow.device("cuda:1"), dtype=flow.int32)
        test_case.assertEqual(output.device, flow.device("cuda:1"))
        test_case.assertEqual(output.dtype, flow.int32)
        test_case.assertTrue(
            np.allclose(input.numpy(), output.numpy(), rtol=0.0001, atol=0.0001)
        )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTo(flow.unittest.TestCase):
    def test_global_tensor_clone(test_case):
        x = flow.ones(
            (4,), placement=flow.placement("cuda", ranks=[0]), sbp=flow.sbp.broadcast
        )
        cloned = x.detach().clone()
        test_case.assertEqual(x.placement, cloned.placement)
        test_case.assertEqual(x.sbp, cloned.sbp)
        cloned_local = cloned.to_local()
        cloned_local[0] = 0
        test_case.assertEqual(cloned_local[0].numpy().item(), 0)
        test_case.assertEqual(x.to_local()[0].numpy().item(), 1)

    def test_global_tensor_to(test_case):
        x = flow.ones(
            (4,), placement=flow.placement("cuda", ranks=[0]), sbp=flow.sbp.broadcast
        )
        cloned = x.to(copy=True)
        test_case.assertEqual(x.placement, cloned.placement)
        test_case.assertEqual(x.sbp, cloned.sbp)
        cloned_local = cloned.to_local()
        cloned_local[0] = 0
        test_case.assertEqual(cloned_local[0].numpy().item(), 0)
        test_case.assertEqual(x.to_local()[0].numpy().item(), 1)

    def test_empty_global_tensor_to(test_case):
        x = flow.ones(
            (0,), placement=flow.placement("cuda", ranks=[0]), sbp=flow.sbp.broadcast
        )
        cloned = x.to(copy=True)
        test_case.assertEqual(x.placement, cloned.placement)
        test_case.assertEqual(x.sbp, cloned.sbp)
        cloned_local = cloned.to_local()
        test_case.assertEqual(tuple(cloned.shape), (0,))
        test_case.assertEqual(tuple(cloned_local.shape), (0,))

    def test_tensor_to_h2d(test_case):
        input = flow.tensor(np.random.randn(2, 3, 4, 5), dtype=flow.float32)
        output = input.to(device=flow.device("cuda"))
        test_case.assertEqual(output.device, flow.device("cuda"))
        test_case.assertTrue(
            np.allclose(input.numpy(), output.numpy(), rtol=0.0001, atol=0.0001)
        )
        gpu_output = output.to(device=flow.device("cuda"))
        test_case.assertEqual(gpu_output.device, flow.device("cuda"))
        test_case.assertTrue(
            np.allclose(input.numpy(), gpu_output.numpy(), rtol=0.0001, atol=0.0001)
        )

    def test_tensor_to_d2h(test_case):
        input = flow.tensor(
            np.random.randn(2, 3, 4, 5), dtype=flow.float32, device=flow.device("cuda")
        )
        output = input.to(device=flow.device("cpu"))
        test_case.assertEqual(output.device, flow.device("cpu"))
        test_case.assertTrue(
            np.allclose(input.numpy(), output.numpy(), rtol=0.0001, atol=0.0001)
        )

    def test_tensor_to_d2d(test_case):
        input = flow.tensor(
            np.random.randn(2, 3, 4, 5), dtype=flow.float32, device=flow.device("cuda")
        )
        output = input.to(device=flow.device("cuda:0"))
        test_case.assertEqual(output.device, flow.device("cuda:0"))
        test_case.assertTrue(
            np.allclose(input.numpy(), output.numpy(), rtol=0.0001, atol=0.0001)
        )

    def test_tensor_to_h2h(test_case):
        input = flow.tensor(np.random.randn(2, 3, 4, 5), dtype=flow.float32)
        output = input.to(device=flow.device("cpu"))
        test_case.assertEqual(output.device, flow.device("cpu"))
        test_case.assertTrue(
            np.allclose(input.numpy(), output.numpy(), rtol=0.0001, atol=0.0001)
        )

    def test_tensor_to_cast(test_case):
        input = flow.tensor(np.random.randn(2, 3, 4, 5), dtype=flow.float32)
        output = input.to(dtype=flow.int)
        test_case.assertEqual(output.dtype, flow.int)

    def test_tensor_to_cast_h2d(test_case):
        input = flow.tensor(np.random.randn(2, 3, 4, 5), dtype=flow.float32)
        output = input.to(device=flow.device("cuda"), dtype=flow.int)
        test_case.assertEqual(output.dtype, flow.int)
        test_case.assertEqual(output.device, flow.device("cuda"))

    def test_tensor_using_tensor(test_case):
        tensor = flow.tensor(np.random.randn(2, 3, 4, 5), device="cuda", dtype=flow.int)
        input = flow.tensor(np.random.randn(2, 3))
        output = input.to(tensor)
        test_case.assertEqual(output.dtype, flow.int)
        test_case.assertEqual(output.device, flow.device("cuda"))

    @autotest(n=5, check_graph=True)
    def test_int_to_args(test_case):
        device_num = random(0, 2).to(int).value()
        x = random_tensor(ndim=4).to(device_num)
        return x

    @autotest(n=5, check_graph=True)
    def test_int_to_kwargs(test_case):
        device_num = random(0, 2).to(int).value()
        x = random_tensor(ndim=4).to(device=device_num)
        return x


if __name__ == "__main__":
    unittest.main()
