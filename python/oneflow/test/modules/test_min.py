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
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestMinModule(flow.unittest.TestCase):
    @autotest(n=5, check_allclose=False, check_graph=True)
    def test_min_reduce_random_dim(test_case):
        device = random_device()
        ndim = random().to(int).value()
        x = random_tensor(ndim=ndim, dim0=random(1, 8))
        y = x.to(device)
        dim = random(-ndim, ndim).to(int).value()
        keep_dims = random_bool().value()
        y = torch.min(x, dim=dim, keepdim=keep_dims)

        # pytorch result is an instance of class 'torch.return_types.min', but oneflow is tuple
        test_case.assertTrue(
            np.allclose(
                y.oneflow[0].detach().cpu().numpy(),
                y.pytorch.values.detach().cpu().numpy(),
                rtol=0.0001,
                atol=1e-05,
            )
        )
        test_case.assertTrue(
            np.allclose(
                y.oneflow[1].detach().cpu().numpy(),
                y.pytorch.indices.detach().cpu().numpy(),
                rtol=0.0001,
                atol=1e-05,
            )
        )

        y.oneflow[0].sum().backward()
        y.pytorch.values.sum().backward()
        test_case.assertTrue(
            np.allclose(
                x.oneflow.grad.detach().cpu().numpy(),
                x.pytorch.grad.detach().cpu().numpy(),
                rtol=0.0001,
                atol=1e-05,
            )
        )

    @autotest(n=5, check_graph=True)
    def test_min_reduce_all_dim(test_case):
        device = random_device()
        ndim = random().to(int).value()
        x = random_tensor(ndim=ndim, dim0=random(1, 8)).to(device)
        return torch.min(x)

    @autotest(n=5, check_graph=True)
    def test_min_elementwise(test_case):
        device = random_device()
        ndim = random().to(int).value()
        dims = [random(1, 8) for _ in range(ndim)]
        x = random_tensor(ndim, *dims).to(device)
        y = random_tensor(ndim, *dims).to(device)
        return torch.min(x, y)

    @autotest(n=5, check_graph=True, check_dtype=True)
    def test_min_elementwise_dtype_promotion(test_case):
        device = random_device()
        ndim = random().to(int).value()
        dims = [random(1, 8) for _ in range(ndim)]
        x = random_tensor(ndim, *dims, dtype=float).to(device)
        y = random_tensor(ndim, *dims, dtype=int).to(device)
        return torch.min(x, y)

    @autotest(n=5, check_graph=True, check_dtype=True)
    def test_min_broadcast_dtype_promotion(test_case):
        device = random_device()
        ndim = random().to(int).value()
        dims = [random(1, 8) for _ in range(ndim)]
        b_dims = [1 for _ in range(ndim)]
        x = random_tensor(ndim, *dims, dtype=float).to(device)
        y = random_tensor(ndim, *b_dims, dtype=int).to(device)
        return torch.min(x, y)

    @autotest(n=3, auto_backward=False)
    def test_min_return_type(test_case):
        x = random_tensor(3, 4)
        result = x.min(1)
        return result.values, result.indices


if __name__ == "__main__":
    unittest.main()
