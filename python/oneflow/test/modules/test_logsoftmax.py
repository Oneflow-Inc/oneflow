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
import oneflow as flow
from automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestSoftmaxModule(flow.unittest.TestCase):
    @autotest()
    def test_against_pytorch(test_case):
        dim = 2
        m = torch.nn.LogSoftmax(dim=1)
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=dim, dim1=random(2, 3), dim2=random(2, 3)).to(
            device
        )
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()
