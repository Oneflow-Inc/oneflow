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


class TestNegHigherDerivative(flow.unittest.TestCase):
    def test_neg_grad_grad(test_case):
        x = random_tensor(ndim=2).requires_grad_(True)
        y = torch.neg(x)
        np_arr = np.random.rand(*x.oneflow.shape)
        init_grad = torch.tensor(np_arr).requires_grad_()

        x_grad = torch.autograd.grad(y, x, init_grad, create_graph=True)[0]
        test_case.assertTrue(
            np.allclose(
                x_grad.pytorch.detach().cpu().numpy(), x_grad.oneflow.detach().numpy()
            )
        )

        init_grad_grad = torch.tensor(np_arr).requires_grad_()
        dgrad = torch.autograd.grad(
            x_grad, init_grad, init_grad_grad, create_graph=False
        )[0]
        test_case.assertTrue(
            np.allclose(
                dgrad.pytorch.detach().cpu().numpy(), dgrad.oneflow.detach().numpy(),
            )
        )


if __name__ == "__main__":
    unittest.main()
