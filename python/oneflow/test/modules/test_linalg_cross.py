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


@flow.unittest.skip_unless_1n1d()
class TestLinalgCross(flow.unittest.TestCase):
    # TODO(peihong): PyTorch 1.10 has no torch.linalg.cross, so uncomment the below code when PyTorch in ci is upgraded to 1.11.
    # @autotest(n=5)
    # def test_linalg_cross_with_random_data(test_case):
    #     device = random_device()
    #     ndim = np.random.randint(2, 6)
    #     shape = list(np.random.randint(16, size=ndim))
    #     index = np.random.randint(ndim)
    #     shape[index] = 3

    #     x = random_tensor(ndim, *shape).to(device)
    #     y = random_tensor(ndim, *shape).to(device)
    #     return torch.linalg.cross(x, y, dim=index)

    # @autotest(n=10)
    # def test_linalg_cross_with_random_data_broadcast(test_case):
    #     device = random_device()
    #     ndim = np.random.randint(3, 6)
    #     shape = list(np.random.randint(16, size=ndim))
    #     indexes = list(np.random.choice(ndim, 3))
    #     shape[indexes[0]] = 3
    #     x_shape = shape
    #     y_shape = shape[:]
    #     x_shape[indexes[1]] = 1
    #     y_shape[indexes[2]] = 1

    #     x = random_tensor(ndim, *x_shape).to(device)
    #     y = random_tensor(ndim, *y_shape).to(device)
    #     return torch.linalg.cross(x, y, dim=indexes[0])

    # @autotest(n=1)
    # def test_linalg_cross_with_random_data_broadcast_different_num_axes(test_case):
    #     device = random_device()
    #     x = random_tensor(4, 4, 5, 3, 5).to(device)
    #     y = random_tensor(3, 1, 3, 5).to(device)
    #     return torch.linalg.cross(x, y, dim=2)

    # @autotest(n=5)
    # def test_linalg_cross_with_random_data_default_dim(test_case):
    #     device = random_device()
    #     ndim = np.random.randint(2, 6)
    #     shape = list(np.random.randint(16, size=ndim))
    #     index = np.random.randint(ndim)
    #     shape[index] = 3

    #     x = random_tensor(ndim, *shape).to(device)
    #     y = random_tensor(ndim, *shape).to(device)
    #     return torch.linalg.cross(x, y)

    @autotest(n=5)
    def test_cross_with_random_data_default_dim(test_case):
        device = random_device()
        ndim = np.random.randint(2, 6)
        shape = list(np.random.randint(16, size=ndim))
        index = np.random.randint(ndim)
        shape[index] = 3

        x = random_tensor(ndim, *shape).to(device)
        y = random_tensor(ndim, *shape).to(device)
        return torch.cross(x, y)


if __name__ == "__main__":
    unittest.main()
