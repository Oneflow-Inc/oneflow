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


class TestAllReduce(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n2d()
    def test_all_reduce_1n2d(test_case):
        np_arr = np.array([[1, 2], [3, 4]])
        of_tensor = flow.tensor(np_arr, device="cuda")
        flow.comm.all_reduce(of_tensor)

        if not torch.distributed.is_initialized():
            dist.init_process_group("gloo")
        torch_tensor = torch.tensor(np_arr)
        dist.all_reduce(torch_tensor)

        test_case.assertTrue(np.allclose(of_tensor.numpy(), torch_tensor.cpu().numpy()))
        dist.destroy_process_group()

    @flow.unittest.skip_unless_2n2d()
    def test_all_reduce_2n2d(test_case):
        np_arr = np.array([[1, 2], [3, 4]])
        tensor = flow.tensor(np_arr, device="cuda")
        flow.comm.all_reduce(tensor)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr * 4))


class TestAllGather(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_all_gather_1n2d(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array([[2, 3], [4, 5]])
        elif flow.env.get_rank() == 1:
            np_arr = np.array([[1, 2], [3, 4]])
        of_input = flow.tensor(np_arr, device="cuda", dtype=flow.int32)
        of_tensor_list = [flow.zeros(np_arr.shape, dtype=flow.int32) for _ in range(2)]
        flow.comm.all_gather(of_tensor_list, of_input)

        if not torch.distributed.is_initialized():
            dist.init_process_group("gloo")
        torch_tensor_list = [
            torch.zeros(np_arr.shape, dtype=torch.int32) for _ in range(2)
        ]
        torch_input = torch.tensor(np_arr, dtype=torch.int32)
        dist.all_gather(torch_tensor_list, torch_input)
        test_case.assertTrue(
            np.allclose(of_tensor_list[0].numpy(), torch_tensor_list[0].cpu().numpy())
        )
        test_case.assertTrue(
            np.allclose(of_tensor_list[1].numpy(), torch_tensor_list[1].cpu().numpy())
        )
        dist.destroy_process_group()


class TestBroadCast(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_broadcast_1n2d(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array([[1, 2], [3, 4]])
        elif flow.env.get_rank() == 1:
            np_arr = np.array([[4, 5], [6, 7]])
        of_tensor = flow.tensor(np_arr, device="cuda", dtype=flow.int32)
        flow.comm.broadcast(of_tensor, 1)

        if not torch.distributed.is_initialized():
            dist.init_process_group("gloo")

        torch_tensor = torch.tensor(np_arr, dtype=torch.int32)
        dist.broadcast(torch_tensor, 1)
        test_case.assertTrue(np.allclose(of_tensor.numpy(), torch_tensor.cpu().numpy()))

        of_tensor = flow.tensor(np_arr, device="cuda", dtype=flow.int32)
        flow.comm.broadcast(of_tensor, 0)
        torch_tensor = torch.tensor(np_arr, dtype=torch.int32)
        dist.broadcast(torch_tensor, 0)
        test_case.assertTrue(np.allclose(of_tensor.numpy(), torch_tensor.cpu().numpy()))
        dist.destroy_process_group()


class TestScatter(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    def test_scatter_1n4d(test_case):
        of_output = flow.tensor([[1, 2], [3, 4]], device="cuda")
        torch_output = torch.tensor([[1, 2], [3, 4]])
        if not torch.distributed.is_initialized():
            dist.init_process_group("gloo")
        if flow.env.get_rank() == 1:
            of_tensor_list = [
                flow.tensor([[5, 6], [7, 8]], device="cuda") + i for i in range(4)
            ]
            flow.comm.scatter(of_output, of_tensor_list, src=1)

            torch_tensor_list = [torch.tensor(x.numpy()) for x in of_tensor_list]
            dist.scatter(torch_output, torch_tensor_list, src=1)
            test_case.assertTrue(np.allclose(of_output.numpy(), torch_output.numpy()))
        else:
            flow.comm.scatter(of_output, src=1)

            dist.scatter(torch_output, src=1)
            test_case.assertTrue(np.allclose(of_output.numpy(), torch_output.numpy()))
        dist.destroy_process_group()


class TestGather(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    def test_gather_1n4d(test_case):
        np_arr = np.array([[1, 2], [3, 4]])
        of_input = flow.tensor(
            np_arr + flow.env.get_rank(), dtype=flow.int32, device="cuda"
        )

        if not torch.distributed.is_initialized():
            dist.init_process_group("gloo")
        torch_input = torch.tensor(np_arr + dist.get_rank(), dtype=torch.int32)

        if flow.env.get_rank() == 1:
            of_tensor_list = [
                flow.zeros(np_arr.shape, dtype=flow.int32, device="cuda")
                for _ in range(4)
            ]
            flow.comm.gather(of_input, gather_list=of_tensor_list, dst=1)

            torch_tensor_list = [
                torch.zeros(np_arr.shape, dtype=torch.int32) for _ in range(4)
            ]
            dist.gather(torch_input, gather_list=torch_tensor_list, dst=1)
            for i in range(4):
                test_case.assertTrue(
                    np.allclose(of_tensor_list[i].numpy(), torch_tensor_list[i].numpy())
                )
        else:
            flow.comm.gather(of_input, dst=1)
            dist.gather(torch_input, dst=1)

        dist.destroy_process_group()


class TestReduce(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_reduce_1n2d(test_case):
        if flow.env.get_rank() == 0:
            np_arr = np.array([[1, 2], [3, 4]])
        elif flow.env.get_rank() == 1:
            np_arr = np.array([[4, 5], [6, 7]])
        of_tensor = flow.tensor(np_arr, device="cuda", dtype=flow.int32)
        flow.comm.reduce(of_tensor, 0)

        if not torch.distributed.is_initialized():
            dist.init_process_group("gloo")
        torch_tensor = torch.tensor(np_arr, dtype=torch.int32)
        dist.reduce(torch_tensor, 0)

        if flow.env.get_rank() == 0:
            test_case.assertTrue(
                np.allclose(of_tensor.numpy(), torch_tensor.cpu().numpy())
            )
        else:
            test_case.assertTrue(
                np.allclose(of_tensor.numpy(), torch_tensor.cpu().numpy())
            )
        dist.destroy_process_group()


class TestAllToAll(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    def test_all_to_all_1n4d(test_case):
        of_input_list = [
            flow.tensor([0, 1], device="cpu") + i * 2 + flow.env.get_rank() * 8
            for i in range(4)
        ]
        of_output_list = [flow.tensor([0, 1], device="cpu") for _ in range(4)]
        flow.comm.all_to_all(of_output_list, of_input_list)

        # only nccl support
        if not torch.distributed.is_initialized():
            dist.init_process_group("nccl")
        torch_input_list = [
            torch.tensor(x.numpy()).to("cuda:{}".format(dist.get_rank()))
            for x in of_input_list
        ]
        torch_output_list = [
            torch.tensor(x.numpy()).to("cuda:{}".format(dist.get_rank()))
            for x in of_output_list
        ]
        dist.all_to_all(torch_output_list, torch_input_list)

        for i in range(len(of_output_list)):
            test_case.assertTrue(
                np.allclose(
                    of_output_list[i].numpy(), torch_output_list[i].cpu().numpy(),
                )
            )
        dist.destroy_process_group()


class TestReduceScatter(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    def test_reduce_scatter_1n4d(test_case):
        of_output = flow.tensor([[0, 0], [0, 0]], device="cpu")
        of_tensor_list = [
            flow.tensor([[1, 2], [3, 4]], device="cpu") + flow.env.get_rank() + i
            for i in range(4)
        ]
        flow.comm.reduce_scatter(of_output, of_tensor_list)

        if not torch.distributed.is_initialized():
            dist.init_process_group("nccl")
        torch_output = torch.tensor([[0, 0], [0, 0]]).to(
            "cuda:{}".format(dist.get_rank())
        )
        torch_tensor_list = [
            torch.tensor(x.numpy()).to("cuda:{}".format(dist.get_rank()))
            for x in of_tensor_list
        ]
        dist.reduce_scatter(torch_output, torch_tensor_list)

        test_case.assertTrue(np.allclose(of_output.numpy(), torch_output.cpu().numpy()))
        dist.destroy_process_group()


@flow.unittest.skip_unless_1n2d()
class TestDocs(flow.unittest.TestCase):
    def test_docs(test_case):
        oneflow.framework.unittest.check_multi_rank_docstr(oneflow.comm.comm_ops)


if __name__ == "__main__":
    unittest.main()
