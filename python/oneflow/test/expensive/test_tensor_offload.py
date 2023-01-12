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


def _test_tensor_offload_d2h(test_case, input, tensor_mem):
    test_case.assertTrue(not input.is_offloaded())

    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    print("cuda", before_used)
    before_id = id(input)

    input.offload()
    test_case.assertTrue(input.is_offloaded())
    test_case.assertEqual(input.device, flow.device("cuda"))
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    after_id = id(input)
    print("cuda to cpu", after_used)
    # Check tensor_mem cuda memory released
    test_case.assertTrue((before_used - after_used) == tensor_mem)
    # if tensor_mem != 0:
    #     test_case.assertTrue(before_used > after_used)
    test_case.assertEqual(before_id, after_id)


def _test_tensor_load_h2d(test_case, input, tensor_mem):
    test_case.assertTrue(input.is_offloaded())

    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    before_id = id(input)

    input.load()
    test_case.assertTrue(not input.is_offloaded())
    test_case.assertEqual(input.device, flow.device("cuda"))
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    after_id = id(input)
    print("cpu to cuda", after_used)
    # Check tensor_mem cuda memory allocated
    test_case.assertTrue((after_used - before_used) == tensor_mem)
    # if tensor_mem != 0:
    #     test_case.assertTrue(after_used > before_used)
    test_case.assertEqual(before_id, after_id)


def _get_tensor_mem(input):
    if input.dim() == 0:
        return 2
    cnt_size = input.element_size() * flow.numel(input)
    return cnt_size / 1024 / 1024


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensorOffload(flow.unittest.TestCase):
    def test_tensor_offload_and_load_float32(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randn(1024, 1024, 100),
            dtype=flow.float32,
            device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = _get_tensor_mem(input)
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    def test_tensor_offload_and_load_float16(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randn(20, 1024, 1024),
            dtype=flow.float16,
            device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = _get_tensor_mem(input)
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    def test_tensor_offload_and_load_int64(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randn(20, 1024, 1024),
            dtype=flow.int64,
            device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = _get_tensor_mem(input)
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    @unittest.skip("0 dim tensor is unstable in CI container mem tests.")
    def test_tensor_offload_and_load_0dim(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randint(1, 10), dtype=flow.float16, device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = _get_tensor_mem(input)
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    def test_tensor_offload_and_load_0size(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randn(0, 1024, 1024),
            dtype=flow.float16,
            device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = 0
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    def test_tensor_offload_and_load_cpu_mem(test_case):
        input = flow.tensor(
            np.random.randn(1024, 1024, 100),
            dtype=flow.float32,
            device=flow.device("cuda"),
        )

        before_used = flow._oneflow_internal.GetCPUMemoryUsed()
        before_id = id(input)
        input.offload()
        after_used = flow._oneflow_internal.GetCPUMemoryUsed()
        after_id = id(input)
        test_case.assertTrue(after_used > before_used)
        test_case.assertEqual(before_id, after_id)

        cur_used = flow._oneflow_internal.GetCPUMemoryUsed()
        before_id = id(input)
        input.load()
        after_used = flow._oneflow_internal.GetCPUMemoryUsed()
        after_id = id(input)
        test_case.assertTrue(after_used < cur_used)
        test_case.assertEqual(before_id, after_id)


if __name__ == "__main__":
    unittest.main()
