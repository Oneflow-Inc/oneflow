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
import re
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def det_random_device():
    cuda_version = flow._oneflow_internal.flags.cuda_version()
    if cuda_version < 11000:  # cuSOLVER is only supported in CUDA 11.0 and above
        return cpu_device()
    else:
        return random_device()


@flow.unittest.skip_unless_1n1d()
class TestLinalgDet(flow.unittest.TestCase):
    @autotest(n=5, rtol=1e-2, auto_backward=False)
    def test_det_3by3_with_random_data(test_case):
        device = det_random_device()
        x = random_tensor(ndim=2, dim0=3, dim1=3, low=-1).to(device)
        return torch.linalg.det(x)

    @autotest(n=5, rtol=1e-2, auto_backward=False)
    def test_det_batch_3by3_with_random_data(test_case):
        device = det_random_device()
        x = random_tensor(ndim=3, dim0=random(), dim1=3, dim2=3, low=-1).to(device)
        return torch.linalg.det(x)

    @autotest(n=5, rtol=1e-2, auto_backward=False)
    def test_det_random_square_with_random_data(test_case):
        device = det_random_device()
        square_dim = random()
        x = random_tensor(ndim=4, dim2=square_dim, dim3=square_dim, low=-1).to(device)
        return torch.linalg.det(x)

    @profile(torch.linalg.det)
    def profile_linalg_det(test_case):
        torch.linalg.det(torch.randn(1, 32, 4, 4))
        torch.linalg.det(torch.randn(16, 32, 4, 4))


if __name__ == "__main__":
    unittest.main()
