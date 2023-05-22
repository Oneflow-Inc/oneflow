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

import numpy as np
import unittest
import os

import oneflow as flow
import oneflow.unittest

import torch
import torch.distributed as dist


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestAllReduce(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_all_reduce_1n2d(test_case):
        np_arr = np.array([[1, 2], [3, 4]])
        tensor = flow.tensor(np_arr, device="cuda")
        flow.comm.all_reduce(tensor)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr * 2))

    @flow.unittest.skip_unless_2n2d()
    def test_all_reduce_2n2d(test_case):
        np_arr = np.array([[1, 2], [3, 4]])
        tensor = flow.tensor(np_arr, device="cuda")
        flow.comm.all_reduce(tensor)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr * 4))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestAllGather(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_all_gather_into_tensor_1n2d(test_case):
        device = "cuda"
        tensor_in = (
            flow.tensor([[1, 2, 3], [4, 5, 6]], dtype=flow.int64, device=device)
            + flow.env.get_rank() * 6
        )
        tensor_out = flow.zeros(4, 3, dtype=flow.int64, device=device)
        flow.comm.all_gather_into_tensor(tensor_out, tensor_in)
        test_case.assertTrue(
            np.allclose(
                tensor_out.numpy(),
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            )
        )

        tensor_out2 = flow.zeros(2, 3, 2, dtype=flow.int64, device=device)
        flow.comm.all_gather_into_tensor(tensor_out2, tensor_in)
        test_case.assertTrue(
            np.allclose(
                tensor_out2.numpy(),
                np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
            )
        )

    @flow.unittest.skip_unless_1n2d()
    def test_all_gather_1n2d(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array([[2, 3], [4, 5]])
        elif flow.env.get_rank() == 1:
            np_arr = np.array([[1, 2], [3, 4]])
        input = flow.tensor(np_arr, device="cuda", dtype=flow.int32)
        tensor_list = [flow.zeros(np_arr.shape, dtype=flow.int32) for _ in range(2)]
        flow.comm.all_gather(tensor_list, input)
        test_case.assertTrue(
            np.allclose(tensor_list[0].numpy(), np.array([[2, 3], [4, 5]]))
        )
        test_case.assertTrue(
            np.allclose(tensor_list[1].numpy(), np.array([[1, 2], [3, 4]]))
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestBroadCast(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_broadcast_1n2d(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array([[1, 2], [3, 4]])
        elif flow.env.get_rank() == 1:
            np_arr = np.array([[4, 5], [6, 7]])
        tensor = flow.tensor(np_arr, device="cuda", dtype=flow.int32)
        flow.comm.broadcast(tensor, 1)
        test_case.assertTrue(np.allclose(tensor.numpy(), np.array([[4, 5], [6, 7]])))

        tensor = flow.tensor(np_arr, device="cuda", dtype=flow.int32)
        flow.comm.broadcast(tensor, 0)
        test_case.assertTrue(np.allclose(tensor.numpy(), np.array([[1, 2], [3, 4]])))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestScatter(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    def test_scatter_1n4d(test_case):
        output = flow.tensor([[1, 2], [3, 4]], device="cuda")
        if flow.env.get_rank() == 1:
            tensor_list = [
                flow.tensor([[5, 6], [7, 8]], device="cuda") + i for i in range(4)
            ]
            flow.comm.scatter(output, tensor_list, src=1)
            test_case.assertTrue(
                np.allclose(output.numpy(), np.array([[6, 7], [8, 9]]))
            )
        else:
            flow.comm.scatter(output, src=1)
            test_case.assertTrue(
                np.allclose(
                    output.numpy(), np.array([[5, 6], [7, 8]]) + flow.env.get_rank()
                )
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGather(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    def test_gather_1n4d(test_case):
        np_arr = np.array([[1, 2], [3, 4]])
        if flow.env.get_rank() == 1:
            input = flow.tensor(
                np_arr + flow.env.get_rank(), device="cuda", dtype=flow.int32
            )
            tensor_list = [flow.zeros(np_arr.shape, dtype=flow.int32) for _ in range(4)]
            flow.comm.gather(input, gather_list=tensor_list, dst=1)
            for i in range(4):
                test_case.assertTrue(
                    np.allclose(tensor_list[i].numpy(), np.array([[1, 2], [3, 4]]) + i)
                )
        else:
            input = flow.tensor(
                np_arr + flow.env.get_rank(), device="cuda", dtype=flow.int32
            )
            flow.comm.gather(input, dst=1)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestReduce(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_reduce_1n2d(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array([[1, 2], [3, 4]])
        elif flow.env.get_rank() == 1:
            np_arr = np.array([[4, 5], [6, 7]])
        tensor = flow.tensor(np_arr, device="cuda", dtype=flow.int32)
        flow.comm.reduce(tensor, 0)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.allclose(tensor.numpy(), np.array([[5, 7], [9, 11]]))
            )
        else:
            test_case.assertTrue(
                np.allclose(tensor.numpy(), np.array([[4, 5], [6, 7]]))
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestAllToAll(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    def test_all_to_all_1n4d(test_case):
        input_list = [
            flow.tensor([0, 1], device="cuda") + i * 2 + flow.env.get_rank() * 8
            for i in range(4)
        ]
        output_list = [flow.tensor([0, 1], device="cuda") for _ in range(4)]
        flow.comm.all_to_all(output_list, input_list)
        for i in range(len(output_list)):
            test_case.assertTrue(
                np.allclose(
                    output_list[i].numpy(),
                    input_list[i].numpy() + (i - flow.env.get_rank()) * 6,
                )
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestReduceScatter(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    def test_reduce_scatter_1n4d(test_case):
        output = flow.tensor([[0, 0], [0, 0]], device="cuda")
        tensor_list = [
            flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_rank() + i
            for i in range(4)
        ]
        flow.comm.reduce_scatter(output, tensor_list)
        test_case.assertTrue(
            np.allclose(output.numpy(), tensor_list[0].numpy() * 4 + 6)
        )

    @flow.unittest.skip_unless_1n2d()
    def test_reduce_scatter_tensor_1n2d(test_case):
        tensor_in = flow.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=flow.int64,
            device="cuda",
        )
        tensor_out = flow.zeros(2, 3, dtype=flow.int64, device="cuda")
        flow.comm.reduce_scatter_tensor(tensor_out, tensor_in)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.allclose(tensor_out.numpy(), np.array([[2, 4, 6], [8, 10, 12]]),)
            )
        else:
            test_case.assertTrue(
                np.allclose(tensor_out.numpy(), np.array([[14, 16, 18], [20, 22, 24]]),)
            )
        tensor_in2 = tensor_in.reshape(2, 3, 2)
        tensor_out2 = flow.zeros(2, 3, dtype=flow.int64, device="cuda")
        flow.comm.reduce_scatter_tensor(tensor_out2, tensor_in2)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.allclose(tensor_out2.numpy(), np.array([[2, 4, 6], [8, 10, 12]]),)
            )
        else:
            test_case.assertTrue(
                np.allclose(
                    tensor_out2.numpy(), np.array([[14, 16, 18], [20, 22, 24]]),
                )
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestDocs(flow.unittest.TestCase):
    def test_docs(test_case):
        oneflow.framework.unittest.check_multi_rank_docstr(oneflow.comm.comm_ops)


if __name__ == "__main__":
    unittest.main()
