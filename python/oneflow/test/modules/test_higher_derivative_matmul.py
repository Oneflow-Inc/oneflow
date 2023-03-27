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
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

import torch as pytorch_origin
import oneflow as oneflow_origin


class TestMatmulHigherDerivative(flow.unittest.TestCase):
    def test_broadcast_matmul_grad_b_grad(test_case):
        broadcast_dims = [
            np.random.randint(2, 10) for _ in range(np.random.randint(1, 3))
        ]
        m = np.random.randint(2, 10)
        n = np.random.randint(2, 10)
        k = np.random.randint(2, 10)

        shape_a = broadcast_dims + [m, k]
        shape_b = [k, n]
        shape_y = broadcast_dims + [m, n]

        a = random_tensor(len(shape_a), *shape_a).requires_grad_(True)
        b = random_tensor(len(shape_b), *shape_b).requires_grad_(True)

        y = torch.matmul(a, b)

        init_grad_a = random_tensor(len(shape_a), *shape_a).requires_grad_(True)
        init_grad_b = random_tensor(len(shape_b), *shape_b).requires_grad_(True)
        init_grad_y = random_tensor(len(shape_y), *shape_y).requires_grad_(True)

        da = torch.autograd.grad(
            outputs=y,
            inputs=a,
            grad_outputs=init_grad_y,
            create_graph=True,
            retain_graph=True,
        )[0]
        test_case.assertTrue(
            np.allclose(
                da.pytorch.detach().cpu().numpy(),
                da.oneflow.detach().numpy(),
                rtol=1e-4,
                atol=1e-5,
            )
        )

        db = torch.autograd.grad(
            outputs=y,
            inputs=b,
            grad_outputs=init_grad_y,
            create_graph=True,
            retain_graph=True,
        )[0]
        test_case.assertTrue(
            np.allclose(
                db.pytorch.detach().cpu().numpy(),
                db.oneflow.detach().numpy(),
                rtol=1e-4,
                atol=1e-5,
            )
        )

        # torch.autograd.grad in autotest does not support inputs/outpus/grad_outputs as a list
        # so use the original pytorch/oneflow module
        dda_pytorch, ddb_pytorch = pytorch_origin.autograd.grad(
            outputs=[da.pytorch, db.pytorch],
            inputs=[a.pytorch, b.pytorch],
            grad_outputs=[init_grad_a.pytorch, init_grad_b.pytorch],
            create_graph=True,
            retain_graph=True,
        )
        dda_oneflow, ddb_oneflow = oneflow_origin.autograd.grad(
            outputs=[da.oneflow, db.oneflow],
            inputs=[a.oneflow, b.oneflow],
            grad_outputs=[init_grad_a.oneflow, init_grad_b.oneflow],
            create_graph=True,
            retain_graph=True,
        )

        test_case.assertTrue(
            np.allclose(
                ddb_pytorch.detach().cpu().numpy(),
                ddb_oneflow.detach().numpy(),
                rtol=1e-4,
                atol=1e-5,
            )
        )
        test_case.assertTrue(
            np.allclose(
                dda_pytorch.detach().cpu().numpy(),
                dda_oneflow.detach().numpy(),
                rtol=1e-4,
                atol=1e-5,
            )
        )

        dgrad_da = torch.autograd.grad(
            outputs=da,
            inputs=init_grad_y,
            grad_outputs=init_grad_a,
            create_graph=True,
            retain_graph=True,
        )[0]
        test_case.assertTrue(
            np.allclose(
                dgrad_da.pytorch.detach().cpu().numpy(),
                dgrad_da.oneflow.detach().numpy(),
                rtol=1e-4,
                atol=1e-5,
            )
        )

        dgrad_db = torch.autograd.grad(
            outputs=db,
            inputs=init_grad_y,
            grad_outputs=init_grad_b,
            create_graph=True,
            retain_graph=True,
        )[0]
        test_case.assertTrue(
            np.allclose(
                dgrad_db.pytorch.detach().cpu().numpy(),
                dgrad_db.oneflow.detach().numpy(),
                rtol=1e-4,
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
