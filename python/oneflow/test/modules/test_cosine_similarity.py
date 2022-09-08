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

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestCosineSimilarity(flow.unittest.TestCase):
    @autotest(n=3)
    def test_cosine_similartiy_module_with_random_data(test_case):
        device = random_device()
        a = random_tensor(ndim=2, dim0=10, dim1=128).to(device)
        b = random_tensor(ndim=2, dim0=10, dim1=128).to(device)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
        cos.train(random())
        output = cos(a, b)
        return output

    @autotest(n=3)
    def test_cosine_similartiy_functional_with_random_data(test_case):
        device = random_device()
        a = random_tensor(ndim=2, dim0=10, dim1=128).to(device)
        b = random_tensor(ndim=2, dim0=10, dim1=128).to(device)
        output = torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-6)
        return output

    @autotest(n=3)
    def test_cosine_similartiy_broadcast_with_random_data(test_case):
        device = random_device()
        a = random_tensor(ndim=2, dim0=10, dim1=128).to(device)
        b = random_tensor(ndim=2, dim0=1, dim1=128).to(device)
        output = torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-6)
        return output

    @autotest(n=3)
    def test_cosine_similartiy_module_with_nonequal_dim_data(test_case):
        device = random_device()
        a = random_tensor(ndim=2, dim0=10, dim1=128).to(device)
        b = random_tensor(ndim=3, dim0=10, dim1=10, dim2=128).to(device)
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6).to(device)
        cos.train(random())
        output = cos(a, b)
        return output

    @unittest.skip(
        reason="https://github.com/Oneflow-Inc/oneflow/issues/8881#issuecomment-1229682453"
    )
    @profile(torch.nn.functional.cosine_similarity)
    def profile_cosine_similarity(test_case):
        input1 = torch.ones(100, 128)
        input2 = torch.ones(100, 128)
        torch.nn.functional.cosine_similarity(input1, input2)
        torch.nn.functional.cosine_similarity(input1, input2, dim=0)


if __name__ == "__main__":
    unittest.main()
