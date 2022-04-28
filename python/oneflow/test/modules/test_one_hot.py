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
from oneflow.test_utils.test_util import GenArgList


from oneflow.test_utils.automated_test_util import *
import oneflow as flow


def _test_one_hot(test_case, device, num_classes, size, on_value, off_value):
    x = np.random.randint(9, size=size)
    input = flow.tensor(x, device=flow.device(device), dtype=flow.int64)
    output = flow.nn.functional.one_hot(input, num_classes, on_value, off_value)
    if num_classes == -1:
        np_outtmp = np.eye(np.max(x) + 1)[x]
    else:
        np_outtmp = np.eye(num_classes)[x]
    np_out = np.where(np_outtmp == 1, on_value, off_value)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-06, 1e-06))


@flow.unittest.skip_unless_1n1d()
class TestOnehot(flow.unittest.TestCase):
    def test_onehot(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_one_hot,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["num_classes"] = [-1, 10, 11]
        arg_dict["size"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["on_value"] = [-1, -0.9, 0, 0.9, 1]
        arg_dict["off_value"] = [-2, -0.5, 0, 0.5, 2]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(auto_backward=False)
    def test_one_hot_scalar(test_case):
        x = torch.tensor(2)
        y = torch.nn.functional.one_hot(x, num_classes=5)
        return y


if __name__ == "__main__":
    unittest.main()
