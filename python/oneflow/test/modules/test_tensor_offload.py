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


def test_tensor_offload_d2h(test_case, input, tensor_mem):
    test_case.assertTrue(not input.is_offloaded())

    flow.cuda.empty_cache()
    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    print("cuda", before_used)

    input.offload()
    test_case.assertTrue(input.is_offloaded())
    test_case.assertEqual(input.device, flow.device("cuda"))
    flow.cuda.empty_cache()
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    print("cuda to cpu", after_used)
    # Check 400M cuda memory released
    test_case.assertTrue((before_used - after_used) == tensor_mem)


def test_tensor_load_h2d(test_case, input, tensor_mem):
    test_case.assertTrue(input.is_offloaded())

    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()

    input.load()
    test_case.assertTrue(not input.is_offloaded())
    test_case.assertEqual(input.device, flow.device("cuda"))
    flow.cuda.empty_cache()
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    print("cpu to cuda", after_used)
    # Check 400M cuda memory allocated
    test_case.assertTrue((after_used - before_used) == tensor_mem)


def get_tensor_mem(input):
    if input.dim() == 0:
        return  2
    shape = input.shape
    tensor_size = shape[0] * shape[1] * shape[2]

    if input.dtype == oneflow.float32:
        return 4 * tensor_size / 1024 / 1024
    elif input.dtype == oneflow.float16:
        return 2 * tensor_size / 1024 / 1024
    elif input.dtype == oneflow.int64:
        return 8 * tensor_size / 1024 / 1024


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
            input_tensor_mem = get_tensor_mem(input)
            # test tensor offload
            test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            test_tensor_load_h2d(test_case, input, input_tensor_mem)

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
            input_tensor_mem = get_tensor_mem(input)
            # test tensor offload
            test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            test_tensor_load_h2d(test_case, input, input_tensor_mem)

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
            input_tensor_mem = get_tensor_mem(input)
            # test tensor offload
            test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    def test_tensor_offload_and_load_0dim(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randint(1,10),
            dtype=flow.float16,
            device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = get_tensor_mem(input)
            # test tensor offload
            test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))


if __name__ == "__main__":
    unittest.main()
