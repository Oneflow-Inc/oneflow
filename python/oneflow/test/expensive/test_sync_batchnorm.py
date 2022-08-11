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

import torch


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestSyncBatchNorm(flow.unittest.TestCase):
    def test_sync_batchnorm3d(test_case):
        os.environ["ONEFLOW_ENABLE_NHWC"] = "0"
        channel = 8
        torch.distributed.init_process_group(backend="gloo", group_name='test_sync_batchnorm3d')

        torch_input = torch.rand(
            32,
            channel,
            32,
            32,
            32,
            requires_grad=True,
            dtype=torch.float32,
            device=f"cuda:{torch.distributed.get_rank()}",
        )
        torch_bn = torch.nn.BatchNorm3d(channel)
        torch_bn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(torch_bn).cuda(
            torch.distributed.get_rank()
        )
        torch_res = torch_bn(torch_input)

        of_input = flow.tensor(
            torch_input.detach().cpu().numpy(), requires_grad=True, device="cuda"
        )
        of_bn = flow.nn.BatchNorm3d(channel)
        of_bn = flow.nn.SyncBatchNorm.convert_sync_batchnorm(of_bn).cuda()
        of_res = of_bn(of_input)

        of_res.sum().backward()
        torch_res.sum().backward()

        test_case.assertTrue(
            np.allclose(torch_res.detach().cpu().numpy(), of_res.numpy(), atol=1e-8)
        )
        test_case.assertTrue(
            np.allclose(
                torch_input.grad.detach().cpu().numpy(),
                of_input.grad.numpy(),
                atol=1e-8,
            )
        )
        torch.distributed.destroy_process_group()

    def test_sync_batchnorm2d(test_case):
        os.environ["ONEFLOW_ENABLE_NHWC"] = "0"
        channel = 8
        torch.distributed.init_process_group(backend="gloo", group_name='test_sync_batchnorm2d')

        torch_input = torch.rand(
            32,
            channel,
            256,
            256,
            requires_grad=True,
            dtype=torch.float32,
            device=f"cuda:{torch.distributed.get_rank()}",
        )
        torch_bn = torch.nn.BatchNorm2d(channel)
        torch_bn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(torch_bn).cuda(
            torch.distributed.get_rank()
        )
        torch_res = torch_bn(torch_input)

        of_input = flow.tensor(
            torch_input.detach().cpu().numpy(), requires_grad=True, device="cuda"
        )
        of_bn = flow.nn.BatchNorm2d(channel)
        of_bn = flow.nn.SyncBatchNorm.convert_sync_batchnorm(of_bn).cuda()
        of_res = of_bn(of_input)

        of_res.sum().backward()
        torch_res.sum().backward()

        test_case.assertTrue(
            np.allclose(torch_res.detach().cpu().numpy(), of_res.numpy(), atol=1e-8)
        )
        test_case.assertTrue(
            np.allclose(
                torch_input.grad.detach().cpu().numpy(),
                of_input.grad.numpy(),
                atol=1e-8,
            )
        )
        torch.distributed.destroy_process_group()

    def test_sync_batchnorm1d(test_case):
        os.environ["ONEFLOW_ENABLE_NHWC"] = "0"
        channel = 8
        torch.distributed.init_process_group(backend="gloo", group_name='test_sync_batchnorm1d')

        torch_input = torch.rand(
            32,
            channel,
            256,
            requires_grad=True,
            dtype=torch.float32,
            device=f"cuda:{torch.distributed.get_rank()}",
        )
        torch_bn = torch.nn.BatchNorm1d(channel)
        torch_bn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(torch_bn).cuda(
            torch.distributed.get_rank()
        )
        torch_res = torch_bn(torch_input)

        of_input = flow.tensor(
            torch_input.detach().cpu().numpy(), requires_grad=True, device="cuda"
        )
        of_bn = flow.nn.BatchNorm1d(channel)
        of_bn = flow.nn.SyncBatchNorm.convert_sync_batchnorm(of_bn).cuda()
        of_res = of_bn(of_input)

        of_res.sum().backward()
        torch_res.sum().backward()

        test_case.assertTrue(
            np.allclose(torch_res.detach().cpu().numpy(), of_res.numpy(), atol=1e-8)
        )
        test_case.assertTrue(
            np.allclose(
                torch_input.grad.detach().cpu().numpy(),
                of_input.grad.numpy(),
                atol=1e-8,
            )
        )
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
