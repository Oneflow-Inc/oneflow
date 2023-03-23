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
import numpy as np

from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import torch as torch_original
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestTruncDivide(flow.unittest.TestCase):
    @autotest(n=5, check_allclose=False, check_graph=True)
    def test_elementwise_trunc_divide_random_data(test_case):
        device = random_device()
        dim0 = random(1, 8)
        dim1 = random(1, 8)
        dim2 = random(1, 8)
        dim3 = random(1, 8)
        x = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dim2=dim2, dim3=dim3).to(device)
        y = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dim2=dim2, dim3=dim3).to(device)

        x.oneflow = x.oneflow.detach().requires_grad_()
        x.pytorch = x.pytorch.detach().requires_grad_()
        y.oneflow = y.oneflow.detach().requires_grad_()
        y.pytorch = y.pytorch.detach().requires_grad_()

        oneflow_out = flow._C.trunc_divide(x.oneflow, y.oneflow)
        torch_out = torch_original.div(x.pytorch, y.pytorch, rounding_mode="trunc")

        test_case.assertTrue(
            np.allclose(
                oneflow_out.detach().cpu().numpy(),
                torch_out.detach().cpu().numpy(),
                rtol=0.0001,
                atol=1e-05,
            )
        )

        oneflow_out.sum().backward()
        torch_out.sum().backward()

        test_case.assertTrue(
            np.allclose(
                x.oneflow.grad.detach().cpu().numpy(),
                x.pytorch.grad.detach().cpu().numpy(),
                rtol=0.0001,
                atol=1e-05,
            )
        )
        test_case.assertTrue(
            np.allclose(
                y.oneflow.grad.detach().cpu().numpy(),
                y.pytorch.grad.detach().cpu().numpy(),
                rtol=0.0001,
                atol=1e-05,
            )
        )

    @autotest(n=5, check_allclose=False, check_graph=True)
    def test_tensor_truncdiv_scalar_random_data(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4,
            dim0=random(1, 8),
            dim1=random(1, 8),
            dim2=random(1, 8),
            dim3=random(1, 8),
        ).to(device)
        x.oneflow = x.oneflow.detach().requires_grad_()
        x.pytorch = x.pytorch.detach().requires_grad_()

        scalar = random().to(float).value()

        oneflow_out = oneflow._C.trunc_divide(x.oneflow, scalar)
        torch_out = torch_original.div(x.pytorch, scalar, rounding_mode="trunc")

        test_case.assertTrue(
            np.allclose(
                oneflow_out.detach().cpu().numpy(),
                torch_out.detach().cpu().numpy(),
                rtol=0.0001,
                atol=1e-5,
            )
        )

        oneflow_out.sum().backward()
        torch_out.sum().backward()

        test_case.assertTrue(
            np.allclose(
                x.oneflow.grad.detach().cpu().numpy(),
                x.pytorch.grad.detach().cpu().numpy(),
                rtol=0.0001,
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
