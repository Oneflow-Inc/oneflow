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


def _test_broadcast_matmul_grad_b_grad_impl(test_case, placement):
    broadcast_dims = [np.random.randint(1, 5) * 8 for _ in range(2)]
    m = np.random.randint(1, 5) * 8
    n = np.random.randint(1, 5) * 8
    k = np.random.randint(1, 5) * 8

    a_shape = broadcast_dims + [m, k]
    b_shape = [k, n]
    y_shape = broadcast_dims + [m, n]

    a = random_tensor(len(a_shape), *a_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    b = random_tensor(len(b_shape), *b_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    init_grad_a = random_tensor(len(a_shape), *a_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    init_grad_b = random_tensor(len(b_shape), *b_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    init_grad_y = random_tensor(len(y_shape), *y_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )

    y = torch.matmul(a, b)

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
            rtol=1e-5,
            atol=1e-2,
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
            rtol=1e-3,
            atol=1e-4,
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
            rtol=1e-3,
            atol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            dda_pytorch.detach().cpu().numpy(),
            dda_oneflow.detach().numpy(),
            rtol=1e-5,
            atol=1e-2,
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
            rtol=1e-5,
            atol=1e-2,
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
            rtol=1e-5,
            atol=1e-2,
        )
    )


class TestGlobalMatmulHigherDerivative(flow.unittest.TestCase):
    @globaltest
    def test_broadcast_matmul_grad_b_grad(test_case):
        for placement in all_placement():
            for i in range(5):
                _test_broadcast_matmul_grad_b_grad_impl(test_case, placement=placement)


if __name__ == "__main__":
    unittest.main()
