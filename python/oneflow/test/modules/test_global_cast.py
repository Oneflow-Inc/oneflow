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
import os

import numpy as np

import oneflow as flow
from oneflow import nn
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList
from oneflow import Tensor
from oneflow.framework.args_tree import ArgsTree


@flow.unittest.skip_unless_1n4d()
class TestGlobalCastModule_1n4d(flow.unittest.TestCase):
    def test_to_global_flatten_hierarchy(test_case):
        x = flow.ones((4, 4), dtype=flow.int32)
        sbp = (flow.sbp.partial_sum,)
        y = x.to_global(
            placement=flow.placement("cpu", ranks=[[0, 1], [2, 3]]),
            sbp=(flow.sbp.partial_sum, flow.sbp.partial_sum),
        )
        placement = flow.placement("cpu", ranks=[0, 1, 2, 3])
        y = y.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_to_global_flatten_hierarchy_cpu_to_gpu(test_case):
        x = flow.ones((4, 4), dtype=flow.int32)
        sbp = (flow.sbp.partial_sum,)
        y = x.to_global(
            placement=flow.placement("cpu", ranks=[[0, 1], [2, 3]]),
            sbp=(flow.sbp.partial_sum, flow.sbp.partial_sum),
        )
        placement = flow.placement("cuda", ranks=[0, 1, 2, 3])
        y = y.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_to_global_flatten_hierarchy_gpu_to_cpu(test_case):
        x = flow.ones((4, 4), dtype=flow.int32)
        sbp = (flow.sbp.partial_sum,)
        y = x.to_global(
            placement=flow.placement("cuda", ranks=[[0, 1], [2, 3]]),
            sbp=(flow.sbp.partial_sum, flow.sbp.partial_sum),
        )
        placement = flow.placement("cpu", ranks=[0, 1, 2, 3])
        y = y.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))

    def test_to_global_broadcast_shape_dtype(test_case):
        if int(os.getenv("RANK")) < 2:
            x = flow.ones((4, 4), dtype=flow.int32)
        else:
            x = flow.zeros((1,), dtype=flow.float)
        placement = flow.placement("cpu", ranks=[0, 1])
        sbp = (flow.sbp.split(0),)
        y = x.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_local_to_global_2d_sbp(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
        sbp = (flow.sbp.split(0), flow.sbp.partial_sum)
        y = x.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_local_to_global_sp_2_bb(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
        sbp = (flow.sbp.split(0), flow.sbp.partial_sum)
        y = x.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        y = y.to_global(sbp=(flow.sbp.broadcast, flow.sbp.broadcast))
        test_case.assertEqual(y.sbp, (flow.sbp.broadcast, flow.sbp.broadcast))
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((8, 4), dtype=np.int32) * 2)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_local_to_global_ps0_2_s0s0(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        x = x * int(os.getenv("RANK"))
        placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
        sbp = (flow.sbp.partial_sum, flow.sbp.split(0))
        y = x.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        sbp = (flow.sbp.split(0), flow.sbp.split(0))
        y = y.to_global(sbp=sbp)
        z = y.to_local()
        if int(os.getenv("RANK")) < 2:
            scale = 2
        else:
            scale = 4
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((2, 4), dtype=np.int32) * scale)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_local_to_global_s0p_2_s0s0(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        x = x * int(os.getenv("RANK"))
        placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
        sbp = (flow.sbp.split(0), flow.sbp.partial_sum)
        y = x.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        sbp = (flow.sbp.split(0), flow.sbp.split(0))
        y = y.to_global(sbp=sbp)
        z = y.to_local()
        if int(os.getenv("RANK")) < 2:
            scale = 1
        else:
            scale = 5
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((2, 4), dtype=np.int32) * scale)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_to_global_loop_broadcast_shape_dtype(test_case):
        if int(os.getenv("RANK")) < 2:
            x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
            a = flow.ones((4, 4), device=flow.device("cpu"), dtype=flow.int32)
        else:
            x = flow.zeros((1,), dtype=flow.float)
            a = flow.zeros((4, 4), device=flow.device("cpu"), dtype=flow.int32)
        placement = flow.placement("cuda", ranks=[0, 1])
        sbp = (flow.sbp.split(0),)
        for i in range(1000):
            if i % 100 == 0:
                print(i)
            y = x.to_global(placement=placement, sbp=sbp)
            b = a.to_global(placement=placement, sbp=flow.sbp.broadcast)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)


@flow.unittest.skip_unless_1n2d()
class TestGlobalCastModule_1n2d(flow.unittest.TestCase):
    def test_to_global_broadcast_shape_dtype(test_case):
        if os.getenv("RANK") == "0":
            x = flow.ones((4, 4), dtype=flow.int32)
        else:
            x = flow.zeros((1,), dtype=flow.float)
        placement = flow.placement("cpu", ranks=[0])
        sbp = (flow.sbp.broadcast,)
        y = x.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))
        test_case.assertEqual(y.dtype, flow.int32)

    def test_local_to_global_broadcast_data(test_case):
        if int(os.getenv("RANK")) == 0:
            x = flow.ones((4, 4), dtype=flow.int32)
        else:
            x = flow.zeros((4, 4), dtype=flow.int32)
        placement = flow.placement("cpu", ranks=[0, 1])
        sbp = (flow.sbp.broadcast,)
        y = x.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(np.array_equal(z.numpy(), np.ones((4, 4), dtype=np.int32)))

    def test_cuda_global_to_global_cpu_s2b(test_case):
        x = flow.ones((4, 4), device=flow.device("cpu"), dtype=flow.int32)
        placement = flow.placement("cpu", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.split(0))
        sbp = (flow.sbp.broadcast,)
        y = y.to_global(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(np.array_equal(z.numpy(), np.ones((8, 4), dtype=np.int32)))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cuda_global_to_global_s2b(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.split(0))
        sbp = (flow.sbp.broadcast,)
        y = y.to_global(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(np.array_equal(z.numpy(), np.ones((8, 4), dtype=np.int32)))

    def test_cuda_global_to_global_cpu_s2p(test_case):
        x = flow.ones((4, 4), device=flow.device("cpu"), dtype=flow.int32)
        placement = flow.placement("cpu", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.split(0))
        sbp = (flow.sbp.partial_sum,)
        y = y.to_global(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        if int(os.getenv("RANK")) == 0:
            test_case.assertTrue(
                np.array_equal(
                    z.numpy(),
                    np.concatenate(
                        (
                            np.ones((4, 4), dtype=np.int32),
                            np.zeros((4, 4), dtype=np.int32),
                        ),
                        axis=0,
                    ),
                )
            )
        else:
            test_case.assertTrue(
                np.array_equal(
                    z.numpy(),
                    np.concatenate(
                        (
                            np.zeros((4, 4), dtype=np.int32),
                            np.ones((4, 4), dtype=np.int32),
                        ),
                        axis=0,
                    ),
                )
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cuda_global_to_global_s2p(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.split(0))
        sbp = (flow.sbp.partial_sum,)
        y = y.to_global(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (8, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        if int(os.getenv("RANK")) == 0:
            test_case.assertTrue(
                np.array_equal(
                    z.numpy(),
                    np.concatenate(
                        (
                            np.ones((4, 4), dtype=np.int32),
                            np.zeros((4, 4), dtype=np.int32),
                        ),
                        axis=0,
                    ),
                )
            )
        else:
            test_case.assertTrue(
                np.array_equal(
                    z.numpy(),
                    np.concatenate(
                        (
                            np.zeros((4, 4), dtype=np.int32),
                            np.ones((4, 4), dtype=np.int32),
                        ),
                        axis=0,
                    ),
                )
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cuda_global_to_global_b2p(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.broadcast)
        sbp = (flow.sbp.partial_sum,)
        y = y.to_global(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        if int(os.getenv("RANK")) == 0:
            test_case.assertTrue(
                np.array_equal(z.numpy(), np.ones((4, 4), dtype=np.int32))
            )
        else:
            test_case.assertTrue(
                np.array_equal(z.numpy(), np.zeros((4, 4), dtype=np.int32))
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cuda_global_to_global_b2s(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.broadcast)
        sbp = (flow.sbp.split(0),)
        y = y.to_global(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(np.array_equal(z.numpy(), np.ones((2, 4), dtype=np.int32)))

    def test_cuda_global_to_global_cpu_p2s(test_case):
        x = flow.ones((4, 4), device=flow.device("cpu"), dtype=flow.int32)
        placement = flow.placement("cpu", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.partial_sum)
        sbp = (flow.sbp.split(0),)
        y = y.to_global(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((2, 4), dtype=np.int32) * 2)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cuda_global_to_global_p2s(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.partial_sum)
        sbp = (flow.sbp.split(0),)
        y = y.to_global(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((2, 4), dtype=np.int32) * 2)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cuda_global_to_global_cuda_h2d(test_case):
        x = flow.ones((4, 4), device=flow.device("cpu"), dtype=flow.int32)
        placement = flow.placement("cpu", ranks=[0, 1])
        cuda_placement = flow.placement("cuda", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.partial_sum)
        y = y.to_global(placement=cuda_placement, sbp=flow.sbp.partial_sum)
        test_case.assertEqual(y.sbp, (flow.sbp.partial_sum,))
        test_case.assertEqual(y.placement, cuda_placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(np.array_equal(z.numpy(), np.ones((4, 4), dtype=np.int32)))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cuda_global_to_global_cpu_p2b(test_case):
        x = flow.ones((4, 4), device=flow.device("cpu"), dtype=flow.int32)
        placement = flow.placement("cpu", ranks=[0, 1])
        cuda_placement = flow.placement("cuda", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.partial_sum)
        import time

        y = y.to_global(placement=cuda_placement, sbp=flow.sbp.partial_sum)
        sbp = (flow.sbp.broadcast,)
        y = y.to_global(placement=cuda_placement, sbp=sbp)
        y = y.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((4, 4), dtype=np.int32) * 2)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cuda_global_to_global_p2b(test_case):
        x = flow.ones((4, 4), device=flow.device("cuda"), dtype=flow.int32)
        placement = flow.placement("cuda", ranks=[0, 1])
        y = x.to_global(placement=placement, sbp=flow.sbp.partial_sum)
        sbp = (flow.sbp.broadcast,)
        y = y.to_global(sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))
        test_case.assertEqual(y.dtype, flow.int32)
        z = y.to_local()
        test_case.assertTrue(
            np.array_equal(z.numpy(), np.ones((4, 4), dtype=np.int32) * 2)
        )


@flow.unittest.skip_unless_1n1d()
class TestGlobalCastModule_1n1d(flow.unittest.TestCase):
    def test_to_global(test_case):
        x = flow.ones((4, 4))
        placement = flow.placement("cpu", ranks=[0])
        sbp = (flow.sbp.broadcast,)
        y = x.to_global(placement=placement, sbp=sbp)
        test_case.assertEqual(y.sbp, sbp)
        test_case.assertEqual(y.placement, placement)
        test_case.assertEqual(tuple(y.shape), (4, 4))


def _test_cpu_p2b_with_random_parameter(test_case, device_list):
    gen_float = np.random.random
    gen_int = np.random.randint
    dtype_list = [
        flow.uint8,
        flow.int8,
        flow.int32,
        flow.int64,
        flow.float32,
        flow.float64,
        flow.double,
    ]

    def choose_shape_and_dtype(seed):
        rng = np.random.default_rng(seed)
        kdtype = rng.integers(low=1, high=len(dtype_list), size=1)
        ndim = rng.integers(low=1, high=4, size=1)
        shape = rng.integers(low=1, high=10, size=ndim)
        return kdtype, shape

    for _ in range(10):
        seed = flow.tensor(gen_int(1, 1000, 1))
        seed = seed.to_global(
            placement=flow.placement.all(seed.device.type), sbp=flow.sbp.broadcast,
        )
        seed = int(seed.to_local().numpy())
        kdtype, shape = choose_shape_and_dtype(seed)
        if kdtype <= 3:
            np_arr = gen_int(1, 10, shape)
        else:
            np_arr = gen_float(shape)
        tensor = flow.tensor(np_arr, device="cpu", dtype=dtype_list[int(kdtype)])
        cpu_tensor = tensor.to_global(
            placement=flow.placement("cpu", device_list), sbp=flow.sbp.partial_sum
        )
        cpu_tensor = cpu_tensor.to_global(sbp=flow.sbp.broadcast)
        tensor = tensor.to("cuda")
        cuda_tensor = tensor.to_global(
            placement=flow.placement("cuda", device_list), sbp=flow.sbp.partial_sum
        )
        cuda_tensor = cuda_tensor.to_global(sbp=flow.sbp.broadcast)
        test_case.assertTrue(
            np.allclose(cpu_tensor.to_local().numpy(), cuda_tensor.to_local().numpy())
        )


def _test_cpu_s2b_with_random_parameter(test_case, device_list):
    gen_float = np.random.random
    gen_int = np.random.randint
    dtype_list = [
        flow.uint8,
        flow.int8,
        flow.int32,
        flow.int64,
        flow.float32,
        flow.float64,
        flow.double,
    ]

    def choose_shape_and_dtype(seed):
        rng = np.random.default_rng(seed)
        kdtype = rng.integers(low=1, high=len(dtype_list), size=1)
        ndim = rng.integers(low=1, high=4, size=1)
        shape = rng.integers(low=1, high=10, size=ndim)
        return kdtype, shape

    for _ in range(10):
        seed = flow.tensor(gen_int(1, 1000, 1))
        seed = seed.to_global(
            placement=flow.placement.all(seed.device.type), sbp=flow.sbp.broadcast,
        )
        seed = int(seed.to_local().numpy())
        kdtype, shape = choose_shape_and_dtype(seed)
        if kdtype <= 3:
            np_arr = gen_int(1, 10, shape)
        else:
            np_arr = gen_float(shape)
        tensor = flow.tensor(np_arr, device="cpu", dtype=dtype_list[int(kdtype)])
        cpu_tensor = tensor.to_global(
            placement=flow.placement("cpu", device_list), sbp=flow.sbp.split(0)
        )
        cpu_tensor = cpu_tensor.to_global(sbp=flow.sbp.broadcast)
        tensor = tensor.to("cuda")
        cuda_tensor = tensor.to_global(
            placement=flow.placement("cuda", device_list), sbp=flow.sbp.split(0)
        )
        cuda_tensor = cuda_tensor.to_global(sbp=flow.sbp.broadcast)
        test_case.assertTrue(
            np.allclose(cpu_tensor.to_local().numpy(), cuda_tensor.to_local().numpy())
        )


def _test_cpu_p2s_with_random_parameter(test_case, device_list):
    gen_float = np.random.random
    gen_int = np.random.randint
    dtype_list = [
        flow.uint8,
        flow.int8,
        flow.int32,
        flow.int64,
        flow.float32,
        flow.float64,
        flow.double,
    ]

    def choose_shape_and_dtype(seed):
        rng = np.random.default_rng(seed)
        kdtype = rng.integers(low=1, high=len(dtype_list), size=1)
        ndim = rng.integers(low=1, high=4, size=1)
        shape = list(rng.integers(low=1, high=5, size=1) * 12) + list(
            rng.integers(low=1, high=10, size=ndim - 1)
        )
        return kdtype, shape

    for _ in range(10):
        seed = flow.tensor(gen_int(1, 1000, 1))
        seed = seed.to_global(
            placement=flow.placement.all(seed.device.type), sbp=flow.sbp.broadcast,
        )
        seed = int(seed.to_local().numpy())
        kdtype, shape = choose_shape_and_dtype(seed)
        if kdtype <= 3:
            np_arr = gen_int(1, 10, shape)
        else:
            np_arr = gen_float(shape)
        tensor = flow.tensor(np_arr, device="cpu", dtype=dtype_list[int(kdtype)])
        cpu_tensor = tensor.to_global(
            placement=flow.placement("cpu", device_list), sbp=flow.sbp.partial_sum
        )
        cpu_tensor = cpu_tensor.to_global(sbp=flow.sbp.split(0))
        tensor = tensor.to("cuda")
        cuda_tensor = tensor.to_global(
            placement=flow.placement("cuda", device_list), sbp=flow.sbp.partial_sum
        )
        cuda_tensor = cuda_tensor.to_global(sbp=flow.sbp.split(0))
        test_case.assertTrue(
            np.allclose(cpu_tensor.to_local().numpy(), cuda_tensor.to_local().numpy())
        )


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalCast(flow.unittest.TestCase):
    def test_cpu_local_tensor_to_gpu_placement(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array([4, 6, 7, 8], dtype=np.float32)
        else:
            np_arr = np.array([0, 0, 0, 0], dtype=np.float32)
        tensor = flow.tensor(np_arr, dtype=flow.float32)
        placement = flow.placement("cuda", [0, 1, 2, 3])
        device = flow.device("cuda")
        global_tensor = tensor.to_global(placement, flow.sbp.broadcast)
        test_case.assertEqual(global_tensor.to_local().device, device)
        test_case.assertEqual(global_tensor.placement, placement)
        test_case.assertTrue(
            np.array_equal(
                global_tensor.to_local().numpy(),
                np.array([4, 6, 7, 8], dtype=np.float32),
            )
        )

    def test_cpu_p2b_with_random_parameter(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_list"] = [[0, 1], [1, 2, 3], [0, 1, 2, 3]]
        for arg in GenArgList(arg_dict):
            _test_cpu_p2b_with_random_parameter(test_case, *arg)

    def test_cpu_s2b_with_random_parameter(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_list"] = [[0, 1], [1, 2, 3], [0, 1, 2, 3]]
        for arg in GenArgList(arg_dict):
            _test_cpu_s2b_with_random_parameter(test_case, *arg)

    def test_cpu_p2s_with_random_parameter(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_list"] = [[0, 1], [1, 2, 3], [0, 1, 2, 3]]
        for arg in GenArgList(arg_dict):
            _test_cpu_p2s_with_random_parameter(test_case, *arg)

    def test_local_to_global_with_wrong_device(test_case):
        np_arr = np.array([4, 6], dtype=np.float32)
        tensor = flow.tensor(
            np_arr,
            device=flow.device("cuda:%d" % ((flow.env.get_rank() + 1) % 4)),
            dtype=flow.float32,
        )
        placement = flow.placement("cuda", ranks=[0, 1, 2, 3])
        device = flow.device("cuda")
        global_tensor = tensor.to_global(placement, flow.sbp.broadcast)
        local_tensor = global_tensor.to_local()
        test_case.assertEqual(local_tensor.device, device)
        test_case.assertEqual(global_tensor.placement, placement)


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalCast_S2S(flow.unittest.TestCase):
    def test_global_to_global_s0_to_s1(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0, 1])
        split0_tensor = tensor.to_global(placement, flow.sbp.split(0))
        split1_tensor = split0_tensor.to_global(placement, flow.sbp.split(1))
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.array_equal(
                    split1_tensor.to_local().numpy(),
                    np.array(
                        [
                            [4.0, 6.0],
                            [6.0, 2.0],
                            [3.0, 7.0],
                            [6.0, 8.0],
                            [2.0, 10.0],
                            [3.0, 9.0],
                            [4.0, 6.0],
                            [6.0, 8.0],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        elif flow.env.get_rank() == 1:
            test_case.assertTrue(
                np.array_equal(
                    split1_tensor.to_local().numpy(),
                    np.array(
                        [
                            [5.0, 20.0],
                            [5.0, 7.0],
                            [5.0, 4.0],
                            [9.0, 4.0],
                            [10.0, 7.0],
                            [10.0, 5.0],
                            [6.0, 9.0],
                            [6.0, 4.0],
                        ],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_s1_to_s0(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0, 1])
        split_tensor = tensor.to_global(placement, flow.sbp.split(0))
        split1_tensor = split_tensor.to_global(placement, flow.sbp.split(1))
        split0_tensor = split1_tensor.to_global(placement, flow.sbp.split(0))
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.array_equal(
                    split0_tensor.to_local().numpy(),
                    np.array(
                        [
                            [4.0, 6.0, 5.0, 20.0],
                            [6.0, 2.0, 5.0, 7.0],
                            [3.0, 7.0, 5.0, 4.0],
                            [6.0, 8.0, 9.0, 4.0],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        elif flow.env.get_rank() == 1:
            test_case.assertTrue(
                np.array_equal(
                    split0_tensor.to_local().numpy(),
                    np.array(
                        [
                            [2.0, 10.0, 10.0, 7.0],
                            [3.0, 9.0, 10.0, 5.0],
                            [4.0, 6.0, 6.0, 9.0],
                            [6.0, 8.0, 6.0, 4.0],
                        ],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_s0_to_s1_cpu(test_case):
        np_arr = np.random.randn(4, 12)

        cuda_device = flow.device("cuda")
        cuda_tensor = flow.tensor(np_arr, device=cuda_device, dtype=flow.float32)
        cuda_placement = flow.placement("cuda", ranks=[1, 3])
        cuda_split0_tensor = cuda_tensor.to_global(cuda_placement, flow.sbp.split(0))
        cuda_split1_tensor = cuda_split0_tensor.to_global(
            cuda_placement, flow.sbp.split(1)
        )

        cpu_device = flow.device("cpu")
        cpu_tensor = flow.tensor(np_arr, device=cpu_device, dtype=flow.float32)
        cpu_placement = flow.placement("cpu", ranks=[1, 3])
        cpu_split0_tensor = cpu_tensor.to_global(cpu_placement, flow.sbp.split(0))
        cpu_split1_tensor = cpu_split0_tensor.to_global(
            cpu_placement, flow.sbp.split(1)
        )

        if flow.env.get_rank() == 0 or flow.env.get_rank() == 1:
            test_case.assertTrue(
                np.array_equal(
                    cuda_split1_tensor.to_local().numpy(),
                    cpu_split1_tensor.to_local().numpy(),
                )
            )

    def test_global_to_global_s1_to_s0_cpu(test_case):
        np_arr = np.random.randn(4, 12)

        cuda_device = flow.device("cuda")
        cuda_tensor = flow.tensor(np_arr, device=cuda_device, dtype=flow.float32)
        cuda_placement = flow.placement("cuda", ranks=[0, 1])
        cuda_split_tensor = cuda_tensor.to_global(cuda_placement, flow.sbp.split(0))
        cuda_split1_tensor = cuda_split_tensor.to_global(
            cuda_placement, flow.sbp.split(1)
        )
        cuda_split0_tensor = cuda_split1_tensor.to_global(
            cuda_placement, flow.sbp.split(0)
        )

        cpu_device = flow.device("cpu")
        cpu_tensor = flow.tensor(np_arr, device=cpu_device, dtype=flow.float32)
        cpu_placement = flow.placement("cpu", ranks=[0, 1])
        cpu_split_tensor = cpu_tensor.to_global(cpu_placement, flow.sbp.split(0))
        cpu_split1_tensor = cpu_split_tensor.to_global(cpu_placement, flow.sbp.split(1))
        cpu_split0_tensor = cpu_split1_tensor.to_global(
            cpu_placement, flow.sbp.split(0)
        )

        if flow.env.get_rank() == 0 or flow.env.get_rank() == 1:
            test_case.assertTrue(
                np.array_equal(
                    cuda_split0_tensor.to_local().numpy(),
                    cpu_split0_tensor.to_local().numpy(),
                )
            )


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalCast_XToB(flow.unittest.TestCase):
    def test_global_to_global_btb_gpu_to_gpu(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]],
                dtype=np.float32,
            )
        elif flow.env.get_rank() == 1:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        elif flow.env.get_rank() == 2:
            np_arr = np.array(
                [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]],
                dtype=np.float32,
            )
        elif flow.env.get_rank() == 3:
            np_arr = np.array(
                [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0, 1])
        global_tensor = tensor.to_global(placement, flow.sbp.broadcast)
        new_placement = flow.placement("cuda", ranks=[0, 1, 2])
        broadcast_tensor = global_tensor.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(broadcast_tensor.placement, new_placement)
        if flow.env.get_rank() != 3:
            test_case.assertTrue(
                np.array_equal(
                    broadcast_tensor.to_local().numpy(),
                    np.array(
                        [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_stb_gpu_to_gpu(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]],
                dtype=np.float32,
            )
        elif flow.env.get_rank() == 1:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        elif flow.env.get_rank() == 2:
            np_arr = np.array(
                [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]],
                dtype=np.float32,
            )
        elif flow.env.get_rank() == 3:
            np_arr = np.array(
                [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0, 1, 2])
        global_tensor = tensor.to_global(placement, flow.sbp.split(0))
        new_placement = flow.placement("cuda", ranks=[0, 1, 2, 3])
        broadcast_tensor = global_tensor.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(broadcast_tensor.placement, new_placement)
        test_case.assertTrue(
            np.array_equal(
                broadcast_tensor.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20],
                        [6, 8, 9, 0],
                        [3, 7, 5, 0],
                        [6, 8, 9, 0],
                        [2, 10, 10, 7],
                        [3, 9, 10, 5],
                        [4, 6, 6, 9],
                        [6, 8, 6, 4],
                        [9, 6, 5, 8],
                        [4, 9, 7, 0],
                        [2, 5, 7, 9],
                        [6, 8, 10, 0],
                    ],
                    dtype=np.float32,
                ),
            )
        )

    def test_global_to_global_ptb_gpu_to_gpu(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]],
                dtype=np.float32,
            )
        elif flow.env.get_rank() == 1:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        elif flow.env.get_rank() == 2:
            np_arr = np.array(
                [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]],
                dtype=np.float32,
            )
        elif flow.env.get_rank() == 3:
            np_arr = np.array(
                [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0, 1, 2])
        global_tensor = tensor.to_global(placement, flow.sbp.partial_sum)
        new_placement = flow.placement("cuda", ranks=[0, 1, 2, 3])
        broadcast_tensor = global_tensor.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(broadcast_tensor.placement, new_placement)
        test_case.assertTrue(
            np.array_equal(
                broadcast_tensor.to_local().numpy(),
                np.array(
                    [
                        [15, 22, 20, 35],
                        [13, 26, 26, 5],
                        [9, 18, 18, 18],
                        [18, 24, 25, 4],
                    ],
                    dtype=np.float32,
                ),
            )
        )


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalCast_1ToN(flow.unittest.TestCase):
    def test_global_to_global_1tob(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0])
        global_tensor = tensor.to_global(placement, flow.sbp.split(0))
        new_placement = flow.placement("cuda", ranks=[0, 1])
        broadcast_tensor = global_tensor.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(broadcast_tensor.placement, new_placement)
        if flow.env.get_rank() < 2:
            test_case.assertTrue(
                np.array_equal(
                    broadcast_tensor.to_local().numpy(),
                    np.array(
                        [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_1top(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", [0])
        global_tensor = tensor.to_global(placement, flow.sbp.split(0))
        new_placement = flow.placement("cuda", ranks=[0, 1])
        partial_sum_tensor = global_tensor.to_global(
            new_placement, flow.sbp.partial_sum
        )
        test_case.assertEqual(partial_sum_tensor.placement, new_placement)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.array_equal(
                    partial_sum_tensor.to_local().numpy(),
                    np.array(
                        [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                        dtype=np.float32,
                    ),
                )
            )
        elif flow.env.get_rank() == 1:
            test_case.assertTrue(
                np.array_equal(
                    partial_sum_tensor.to_local().numpy(),
                    np.array(
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_1tos(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0])
        global_tensor = tensor.to_global(placement, flow.sbp.split(0))
        new_placement = flow.placement("cuda", ranks=[0, 1, 2, 3])
        split_tensor = global_tensor.to_global(new_placement, flow.sbp.split(0))
        test_case.assertEqual(split_tensor.placement, new_placement)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.array_equal(
                    split_tensor.to_local().numpy(),
                    np.array([[4, 6, 5, 20]], dtype=np.float32,),
                )
            )
        elif flow.env.get_rank() == 1:
            test_case.assertTrue(
                np.array_equal(
                    split_tensor.to_local().numpy(),
                    np.array([[6, 2, 5, 7]], dtype=np.float32,),
                )
            )
        elif flow.env.get_rank() == 2:
            test_case.assertTrue(
                np.array_equal(
                    split_tensor.to_local().numpy(),
                    np.array([[3, 7, 5, 4]], dtype=np.float32,),
                )
            )
        elif flow.env.get_rank() == 3:
            test_case.assertTrue(
                np.array_equal(
                    split_tensor.to_local().numpy(),
                    np.array([[6, 8, 9, 4]], dtype=np.float32,),
                )
            )


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalCast_NTo1(flow.unittest.TestCase):
    def test_global_to_global_bt1(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0, 1])
        global_tensor = tensor.to_global(placement, flow.sbp.broadcast)
        new_placement = flow.placement("cuda", ranks=[0])
        broadcast_tensor = global_tensor.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(broadcast_tensor.placement, new_placement)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.array_equal(
                    broadcast_tensor.to_local().numpy(),
                    np.array(
                        [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_st1(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0, 1])
        global_tensor = tensor.to_global(placement, flow.sbp.split(0))
        new_placement = flow.placement("cuda", ranks=[0])
        partial_sum_tensor = global_tensor.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(partial_sum_tensor.placement, new_placement)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.array_equal(
                    partial_sum_tensor.to_local().numpy(),
                    np.array(
                        [
                            [4, 6, 5, 20],
                            [6, 2, 5, 7],
                            [3, 7, 5, 4],
                            [6, 8, 9, 4],
                            [2, 10, 10, 7],
                            [3, 9, 10, 5],
                            [4, 6, 6, 9],
                            [6, 8, 6, 4],
                        ],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_pt1(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0, 1])
        global_tensor = tensor.to_global(placement, flow.sbp.partial_sum)
        new_placement = flow.placement("cuda", ranks=[0])
        partial_sum_tensor = global_tensor.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(partial_sum_tensor.placement, new_placement)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.array_equal(
                    partial_sum_tensor.to_local().numpy(),
                    np.array(
                        [
                            [6, 16, 15, 27],
                            [9, 11, 15, 12],
                            [7, 13, 11, 13],
                            [12, 16, 15, 8],
                        ],
                        dtype=np.float32,
                    ),
                )
            )


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalCast_1To1(flow.unittest.TestCase):
    def test_global_to_global_1to1_gpu_to_gpu(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        local_tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[3])
        x = local_tensor.to_global(placement, flow.sbp.split(0))
        new_placement = flow.placement("cuda", ranks=[2])
        y = x.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(y.placement, new_placement)
        if flow.env.get_rank() == 2:
            test_case.assertTrue(
                np.array_equal(
                    y.to_local().numpy(),
                    np.array(
                        [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_1to1_cpu_to_cpu(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cpu")
        local_tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cpu", ranks=[0])
        x = local_tensor.to_global(placement, flow.sbp.split(0))
        new_placement = flow.placement("cpu", ranks=[2])
        y = x.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(y.placement, new_placement)
        if flow.env.get_rank() == 2:
            test_case.assertTrue(
                np.array_equal(
                    y.to_local().numpy(),
                    np.array(
                        [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_1to1_gpu_to_cpu(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        local_tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", ranks=[0])
        x = local_tensor.to_global(placement, flow.sbp.split(0))
        new_placement = flow.placement("cpu", ranks=[3])
        y = x.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(y.placement, new_placement)
        if flow.env.get_rank() == 3:
            test_case.assertTrue(
                np.array_equal(
                    y.to_local().numpy(),
                    np.array(
                        [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                        dtype=np.float32,
                    ),
                )
            )

    def test_global_to_global_1to1_cpu_to_gpu(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 2, 5, 7], [3, 7, 5, 4], [6, 8, 9, 4]],
                dtype=np.float32,
            )
        else:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        device = flow.device("cpu")
        local_tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cpu", ranks=[1])
        x = local_tensor.to_global(placement, flow.sbp.split(0))
        new_placement = flow.placement("cuda", ranks=[3])
        y = x.to_global(new_placement, flow.sbp.broadcast)
        test_case.assertEqual(y.placement, new_placement)
        if flow.env.get_rank() == 3:
            test_case.assertTrue(
                np.array_equal(
                    y.to_local().numpy(),
                    np.array(
                        [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                        dtype=np.float32,
                    ),
                )
            )


class GraphTestModel(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, x):
        return self.model(x)


@flow.unittest.skip_unless_1n2d()
class TestToGlobalAndLocal(flow.unittest.TestCase):
    placement = flow.placement("cpu", ranks=[0, 1])
    sbp = None
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    local_graph_model = GraphTestModel(model)
    global_graph_model = None

    def __all_global(test_case, input, placement, sbp):
        if type(input) == Tensor:
            test_case.assertTrue(input.is_global)
            # check placement
            test_case.assertEqual(placement.type, input.placement.type)
            test_case.assertListEqual(
                list(placement.ranks), list(input.placement.ranks)
            )
            # check sbp
            test_case.assertTupleEqual(sbp, input.sbp)
        elif isinstance(input, (dict, tuple, list)):
            node_tree = ArgsTree(input)
            for node in node_tree.iter_nodes():
                if isinstance(node, Tensor):
                    test_case.assertTrue(node.is_global)
                    # check placement
                    test_case.assertEqual(placement.type, node.placement.type)
                    test_case.assertListEqual(
                        list(placement.ranks), list(node.placement.ranks)
                    )
                    # check sbp
                    test_case.assertTupleEqual(sbp, node.sbp)

    def __all_local(test_case, input):
        if type(input) == Tensor:
            test_case.assertFalse(input.is_global)
        elif isinstance(input, (dict, tuple, list)):
            node_tree = ArgsTree(input)
            for node in node_tree.iter_nodes():
                if isinstance(node, Tensor):
                    test_case.assertFalse(node.is_global)

    def _test_any_input(test_case):
        tensor = flow.zeros((3, 4))
        tensor_list = [flow.tensor([1, 2, 3]), flow.randn((2, 3, 4))]
        tensor_tuple = (flow.zeros((2, 2)), flow.ones((2, 3)), flow.randn((3, 5)))
        tensor_dict = {"tensor": tensor, "tensor_lt": tensor_list}
        random_combination = [
            None,
            1,
            "test_str",
            tensor,
            tensor_list,
            tensor_tuple,
            tensor_dict,
        ]

        inputs = [
            None,
            100,
            "test_str",
            tensor,
            tensor_list,
            tensor_tuple,
            tensor_dict,
            random_combination,
        ]
        global_inputs = []
        for i in inputs:
            ret = flow.utils.global_view.to_global(
                i,
                placement=TestToGlobalAndLocal.placement,
                sbp=TestToGlobalAndLocal.sbp,
            )
            test_case.__all_global(
                ret,
                placement=TestToGlobalAndLocal.placement,
                sbp=TestToGlobalAndLocal.sbp,
            )
            global_inputs.append(ret)

        for i in global_inputs:
            ret = flow.utils.global_view.to_local(i)
            test_case.__all_local(ret)

    def _test_any_input_get_sbp_func(test_case):
        def __get_sbp(input, tensor):
            return TestToGlobalAndLocal.sbp

        tensor = flow.zeros((3, 4))
        tensor_list = [flow.tensor([1, 2, 3]), flow.randn((2, 3, 4))]
        tensor_tuple = (flow.zeros((2, 2)), flow.ones((2, 3)), flow.randn((3, 5)))
        tensor_dict = {"tensor": tensor, "tensor_lt": tensor_list}
        random_combination = [
            None,
            1,
            "test_str",
            tensor,
            tensor_list,
            tensor_tuple,
            tensor_dict,
        ]

        inputs = [
            None,
            100,
            "test_str",
            tensor,
            tensor_list,
            tensor_tuple,
            tensor_dict,
            random_combination,
        ]
        global_inputs = []
        for i in inputs:
            ret = flow.utils.global_view.to_global(
                i, placement=TestToGlobalAndLocal.placement, sbp=__get_sbp,
            )
            test_case.__all_global(
                ret,
                placement=TestToGlobalAndLocal.placement,
                sbp=TestToGlobalAndLocal.sbp,
            )
            global_inputs.append(ret)

        for i in global_inputs:
            ret = flow.utils.global_view.to_local(i)
            test_case.__all_local(ret)

    def _test_tensor_to_global(test_case):
        local_tensor = flow.ones((3, 4))

        # local tensor -> global tensor
        global_tensor = flow.utils.global_view.to_global(
            local_tensor,
            placement=TestToGlobalAndLocal.placement,
            sbp=TestToGlobalAndLocal.sbp,
        )
        test_case.assertTrue(global_tensor.is_global)

        # global tensor -> global tensor
        global_tensor = flow.utils.global_view.to_global(
            global_tensor,
            placement=TestToGlobalAndLocal.placement,
            sbp=TestToGlobalAndLocal.sbp,
        )
        test_case.assertTrue(global_tensor.is_global)

        # passing no placement and sbp
        with test_case.assertRaises(ValueError):
            global_tensor = flow.utils.global_view.to_global(
                local_tensor, placement=None, sbp=None
            )

        # wrong sbp type
        with test_case.assertRaises(TypeError):
            global_tensor = flow.utils.global_view.to_global(
                local_tensor,
                placement=TestToGlobalAndLocal.placement,
                sbp=(TestToGlobalAndLocal.sbp, 0),
            )

    def _test_tensor_to_local(test_case):
        # global tensor -> local tensor
        global_tensor = flow.ones(
            (3, 4),
            placement=TestToGlobalAndLocal.placement,
            sbp=TestToGlobalAndLocal.sbp,
        )
        local_tensor = flow.utils.global_view.to_local(global_tensor)
        test_case.assertFalse(local_tensor.is_global)

    def __test_state_dict_to_global(test_case, local_state_dict):
        # local state dict -> global state dict
        global_state_dict = flow.utils.global_view.to_global(
            local_state_dict,
            placement=TestToGlobalAndLocal.placement,
            sbp=TestToGlobalAndLocal.sbp,
        )
        test_case.__all_global(
            global_state_dict,
            placement=TestToGlobalAndLocal.placement,
            sbp=TestToGlobalAndLocal.sbp,
        )

        # global state dict -> global state dict
        global_state_dict = flow.utils.global_view.to_global(
            global_state_dict,
            placement=TestToGlobalAndLocal.placement,
            sbp=TestToGlobalAndLocal.sbp,
        )
        test_case.__all_global(
            global_state_dict,
            placement=TestToGlobalAndLocal.placement,
            sbp=TestToGlobalAndLocal.sbp,
        )

    def __test_state_dict_to_local(test_case, global_state_dict):
        # global state dict -> local state dict
        local_state_dict = flow.utils.global_view.to_local(global_state_dict)
        test_case.__all_local(local_state_dict)

        # local input, display warning
        local_state_dict = flow.utils.global_view.to_local(local_state_dict)

    def _test_eagar_state_dict(test_case):
        test_case.__test_state_dict_to_global(TestToGlobalAndLocal.model.state_dict())
        global_model = TestToGlobalAndLocal.model.to_global(
            placement=TestToGlobalAndLocal.placement, sbp=TestToGlobalAndLocal.sbp
        )
        test_case.__test_state_dict_to_local(global_model.state_dict())

    def _test_graph_state_dict(test_case):
        test_case.__test_state_dict_to_global(
            TestToGlobalAndLocal.local_graph_model.state_dict()
        )
        test_case.__test_state_dict_to_local(
            TestToGlobalAndLocal.global_graph_model.state_dict()
        )

    def test_to_global_local(test_case):
        sbp_types = [
            (flow.sbp.broadcast,),
            (flow.sbp.split(0),),
            (flow.sbp.partial_sum,),
        ]
        for sbp in sbp_types:
            TestToGlobalAndLocal.sbp = sbp
            TestToGlobalAndLocal.global_graph_model = GraphTestModel(
                TestToGlobalAndLocal.model.to_global(
                    placement=TestToGlobalAndLocal.placement, sbp=sbp
                )
            )
            test_case._test_any_input()
            test_case._test_any_input_get_sbp_func()
            test_case._test_tensor_to_global()
            test_case._test_tensor_to_local()
            test_case._test_eagar_state_dict()
            test_case._test_graph_state_dict()


if __name__ == "__main__":
    unittest.main()
