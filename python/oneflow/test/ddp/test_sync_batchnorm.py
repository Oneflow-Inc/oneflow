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
from sync_batchnorm_test_util import ensure_datas


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestSyncBatchNorm(flow.unittest.TestCase):
    def test_sync_batchnorm3d(test_case):
        data_path = ensure_datas()
        os.environ["ONEFLOW_ENABLE_NHWC"] = "0"
        channel = 8
        input_np = np.load(
            f"{data_path}/sync_bn3d_nchw_input_rank{flow.env.get_rank()}.npy"
        )
        torch_out = np.load(
            f"{data_path}/sync_bn3d_nchw_torch_output_rank{flow.env.get_rank()}.npy"
        )
        torch_grad = np.load(
            f"{data_path}/sync_bn3d_nchw_torch_grad_rank{flow.env.get_rank()}.npy"
        )

        of_input = flow.tensor(input_np, requires_grad=True, device="cuda")
        of_bn = flow.nn.BatchNorm3d(channel)
        of_bn = flow.nn.SyncBatchNorm.convert_sync_batchnorm(of_bn).cuda()
        of_res = of_bn(of_input)
        of_res.sum().backward()

        test_case.assertTrue(np.allclose(torch_out, of_res.numpy(), atol=1e-8))
        test_case.assertTrue(np.allclose(torch_grad, of_input.grad.numpy(), atol=1e-8,))

    def test_sync_batchnorm2d(test_case):
        data_path = ensure_datas()
        os.environ["ONEFLOW_ENABLE_NHWC"] = "0"
        channel = 8
        input_np = np.load(
            f"{data_path}/sync_bn2d_nchw_input_rank{flow.env.get_rank()}.npy"
        )
        torch_out = np.load(
            f"{data_path}/sync_bn2d_nchw_torch_output_rank{flow.env.get_rank()}.npy"
        )
        torch_grad = np.load(
            f"{data_path}/sync_bn2d_nchw_torch_grad_rank{flow.env.get_rank()}.npy"
        )

        of_input = flow.tensor(input_np, requires_grad=True, device="cuda")
        of_bn = flow.nn.BatchNorm2d(channel)
        of_bn = flow.nn.SyncBatchNorm.convert_sync_batchnorm(of_bn).cuda()
        of_res = of_bn(of_input)
        of_res.sum().backward()

        test_case.assertTrue(np.allclose(torch_out, of_res.numpy(), atol=1e-8))
        test_case.assertTrue(np.allclose(torch_grad, of_input.grad.numpy(), atol=1e-8,))

    def test_sync_batchnorm1d(test_case):
        data_path = ensure_datas()
        os.environ["ONEFLOW_ENABLE_NHWC"] = "0"
        channel = 8
        input_np = np.load(
            f"{data_path}/sync_bn2d_nchw_input_rank{flow.env.get_rank()}.npy"
        )
        torch_out = np.load(
            f"{data_path}/sync_bn2d_nchw_torch_output_rank{flow.env.get_rank()}.npy"
        )
        torch_grad = np.load(
            f"{data_path}/sync_bn2d_nchw_torch_grad_rank{flow.env.get_rank()}.npy"
        )

        of_input = flow.tensor(input_np, requires_grad=True, device="cuda")
        of_bn = flow.nn.BatchNorm1d(channel)
        of_bn = flow.nn.SyncBatchNorm.convert_sync_batchnorm(of_bn).cuda()
        of_res = of_bn(of_input)
        of_res.sum().backward()

        test_case.assertTrue(np.allclose(torch_out, of_res.numpy(), atol=1e-8))
        test_case.assertTrue(np.allclose(torch_grad, of_input.grad.numpy(), atol=1e-8,))


if __name__ == "__main__":
    unittest.main()
