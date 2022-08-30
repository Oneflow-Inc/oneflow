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
class TestGluModule(flow.unittest.TestCase):
    @autotest(n=5, check_graph=True)
    def test_glu_module_with_random_data(test_case):
        device = random_device()
        dim = random(-3, 3).to(int)
        m = torch.nn.functional.glu
        x = random_tensor(ndim=3, dim0=2, dim1=4, dim2=6).to(device)
        y = m(x, dim)
        return y

    @autotest(n=5, check_graph=True)
    def test_glu_module_with_random_data(test_case):
        device = random_device()
        m = torch.nn.GLU()
        m.train(random())
        m.to(device)
        x = random_tensor(ndim=3, dim0=2, dim1=4, dim2=6).to(device)
        y = m(x)
        return y

    @profile(torch.nn.functional.glu)
    def profile_glu(test_case):
        input = torch.ones(1000, 1000)
        torch.nn.functional.glu(input)


if __name__ == "__main__":
    unittest.main()
