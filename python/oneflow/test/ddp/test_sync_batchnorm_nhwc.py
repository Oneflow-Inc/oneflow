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
import datetime
import numpy as np

import oneflow as flow
import oneflow.unittest

import torch


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestSyncBatchNormChannelLast(flow.unittest.TestCase):
    def test_sync_batchnorm2d_nhwc(test_case):
        os.environ["ONEFLOW_ENABLE_NHWC"] = "1"

        channel = 8
        of_input = flow.rand(
            32, 256, 256, channel, requires_grad=True, dtype=flow.float32, device="cuda"
        )
        of_bn = flow.nn.BatchNorm2d(channel)
        of_bn = flow.nn.SyncBatchNorm.convert_sync_batchnorm(of_bn).cuda()
        of_res = of_bn(of_input)

        torch.distributed.init_process_group(
            backend="gloo", group_name="test_sync_batchnorm2d_nhwc", timeout=datetime.timedelta(seconds=3600)
        )
        torch_input = torch.tensor(
            of_input.numpy(),
            requires_grad=True,
            device=f"cuda:{torch.distributed.get_rank()}",
        )
        torch_input1 = torch_input.permute(0, 3, 1, 2)
        torch_input1 = torch_input1.to(memory_format=torch.channels_last)
        torch_bn = torch.nn.BatchNorm2d(channel)
        torch_bn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(torch_bn).cuda(
            torch.distributed.get_rank()
        )
        torch_res = torch_bn(torch_input1)

        of_res.sum().backward()
        torch_res.sum().backward()

        test_case.assertTrue(
            np.allclose(
                torch_res.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy(),
                of_res.numpy(),
                atol=1e-5,
            )
        )
        test_case.assertTrue(
            np.allclose(
                torch_input.grad.detach().cpu().numpy(),
                of_input.grad.numpy(),
                atol=1e-5,
            )
        )
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
