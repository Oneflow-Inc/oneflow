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


def _global_slice_grad_grad_impl(test_case, placement, sbp):
    x = (
        random_tensor(ndim=3, dim0=8, dim1=8, dim2=8)
        .to_global(placement=placement, sbp=sbp)
        .requires_grad_(True)
    )
    init_grad = (
        random_tensor(ndim=3, dim0=8, dim1=8, dim2=4)
        .to_global(placement=placement, sbp=sbp)
        .requires_grad_(True)
    )
    init_grad_grad = (
        random_tensor(ndim=3, dim0=8, dim1=8, dim2=8)
        .to_global(placement=placement, sbp=sbp)
        .requires_grad_(True)
    )

    y = x[:, :, 2:6]

    x_grad = torch.autograd.grad(y, x, init_grad, create_graph=True)[0]
    test_case.assertTrue(
        np.allclose(
            x_grad.pytorch.detach().cpu().numpy(), x_grad.oneflow.detach().numpy()
        )
    )

    dgrad = torch.autograd.grad(x_grad, init_grad, init_grad_grad, create_graph=False)[
        0
    ]
    test_case.assertTrue(
        np.allclose(
            dgrad.pytorch.detach().cpu().numpy(), dgrad.oneflow.detach().numpy(),
        )
    )


class TestGlobalSliceHigherDerivative(flow.unittest.TestCase):
    @globaltest
    def test_global_slice_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_slice_grad_grad_impl(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
