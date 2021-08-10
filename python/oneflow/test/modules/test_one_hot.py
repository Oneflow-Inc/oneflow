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
from collections import OrderedDict

import numpy as np
from automated_test_util import *
from test_util import GenArgList
import oneflow as flow

def _test_onehot(test_case, device, num_classes):
    input1 = flow.Tensor(np.arange(1, 9).reshape((1, 2, 4)), device=flow.device(device) ,dtype=flow.int64)
    output = flow.one_hot(input1, num_classes=num_classes)
    input2 = torch.arange(1, 9).reshape((1, 2, 4))
    np_out = torch.nn.functional.one_hot(input2, num_classes)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestOnehot(flow.unittest.TestCase):

    @autotest(auto_backward=False)
    def test_one_hot_with_random_data(test_case):
        device = random_device()
        hight = random(1, 6).to(int)
        input = random_pytorch_tensor(high = hight, dtype = int).to(device)
        num_classes = random(low = hight + 1, high = hight + 6).to(int).value()
        y = torch.nn.functional.one_hot(input, num_classes=num_classes)
        return y
    

if __name__ == "__main__":
    unittest.main()
