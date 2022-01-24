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
class TestFunctionalNormalize(flow.unittest.TestCase):
    @autotest()
    def test_functional_normalize(test_case):
        device = random_device()
        ndim = random(low=2)

        shape = list(random_tensor(ndim).value().shape)
        dim = random(low=0, high=ndim).to(int).value()
        shape[dim] = random(low=2, high=8).to(int).value()
        shape = tuple(shape)

        x = random_pytorch_tensor(len(shape), *shape).to(device)
        y = torch.nn.functional.normalize(x, oneof(2, 3, 4), dim, 1e-12)

        return y


if __name__ == "__main__":
    unittest.main()
