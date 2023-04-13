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


def random_index(dim):
    start = np.random.choice(list(range(dim)))
    stop = np.random.choice(list(range(1, dim + 1)))
    if start >= stop:
        start, stop = stop - 1, start + 1
    step = np.random.randint(1, dim)
    return f"{start}:{stop}:{step}"


def random_slice(dim_vec):
    slice_index = ", ".join(random_index(dim) for dim in dim_vec)
    return slice_index


def _test_slice_grad_grad_impl(test_case):
    ndim = np.random.randint(2, 5)
    x_shape = [np.random.randint(3, 8) for _ in range(ndim)]
    x = random_tensor(len(x_shape), *x_shape).requires_grad_(True)

    slice_index = random_slice(x_shape)
    y = eval(f"x[{slice_index}]")

    init_grad = random_tensor(len(y.oneflow.shape), *y.oneflow.shape).requires_grad_()
    x_grad = torch.autograd.grad(y, x, init_grad, create_graph=True)[0]
    test_case.assertTrue(
        np.allclose(
            x_grad.pytorch.detach().cpu().numpy(), x_grad.oneflow.detach().numpy()
        )
    )

    init_grad_grad = random_tensor(
        len(x_grad.oneflow.shape), *x_grad.oneflow.shape
    ).requires_grad_()
    dgrad = torch.autograd.grad(x_grad, init_grad, init_grad_grad, create_graph=False)[
        0
    ]
    test_case.assertTrue(
        np.allclose(
            dgrad.pytorch.detach().cpu().numpy(), dgrad.oneflow.detach().numpy(),
        )
    )


class TestSliceHigherDerivative(flow.unittest.TestCase):
    def test_slice_grad_grad(test_case):
        for i in range(10):
            _test_slice_grad_grad_impl(test_case)


if __name__ == "__main__":
    unittest.main()
