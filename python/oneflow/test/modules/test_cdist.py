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
import random as random_utils

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestCDist(flow.unittest.TestCase):
    @autotest(n=2, check_graph=True)
    def test_zero_cdist(test_case):
        device = random_device()
        dim0 = random()
        dim2 = random()
        mode = random_utils.choice(
            [
                "use_mm_for_euclid_dist_if_necessary",
                "use_mm_for_euclid_dist",
                "donot_use_mm_for_euclid_dist",
            ]
        )
        x1 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        x2 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        return torch.cdist(x1, x2, p=0, compute_mode=mode)

    @autotest(n=2, check_graph=True)
    def test_one_cdist(test_case):
        device = random_device()
        dim0 = random()
        dim2 = random()
        mode = random_utils.choice(
            [
                "use_mm_for_euclid_dist_if_necessary",
                "use_mm_for_euclid_dist",
                "donot_use_mm_for_euclid_dist",
            ]
        )
        x1 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        x2 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        return torch.cdist(x1, x2, p=1, compute_mode=mode)

    @autotest(n=2, check_graph=True)
    def test_two_cdist(test_case):
        device = random_device()
        dim0 = random()
        dim2 = random()
        mode = random_utils.choice(
            [
                "use_mm_for_euclid_dist_if_necessary",
                "use_mm_for_euclid_dist",
                "donot_use_mm_for_euclid_dist",
            ]
        )
        x1 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        x2 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        return torch.cdist(x1, x2, p=2, compute_mode=mode)

    @autotest(n=2, check_graph=True)
    def test_infi_cdist(test_case):
        device = random_device()
        dim0 = random()
        dim2 = random()
        mode = random_utils.choice(
            [
                "use_mm_for_euclid_dist_if_necessary",
                "use_mm_for_euclid_dist",
                "donot_use_mm_for_euclid_dist",
            ]
        )
        x1 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        x2 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        return torch.cdist(x1, x2, p=float("inf"), compute_mode=mode)

    @autotest(n=5, check_graph=True)
    def test_random_p_cdist(test_case):
        device = random_device()
        dim0 = random()
        dim2 = random()
        mode = random_utils.choice(
            [
                "use_mm_for_euclid_dist_if_necessary",
                "use_mm_for_euclid_dist",
                "donot_use_mm_for_euclid_dist",
            ]
        )
        x1 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        x2 = random_tensor(ndim=3, dim0=dim0, dim1=random(), dim2=dim2).to(device)
        p = random(0, 4).to(float)
        return torch.cdist(x1, x2, p=p, compute_mode=mode)


if __name__ == "__main__":
    unittest.main()
