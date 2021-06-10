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

import oneflow.experimental as flow


from test_util import GenArgList


def _test_adaptive_avgpool2d_forward(test_case, device):
    input = flow.Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32), dtype=flow.float32, device=flow.device(device))
    m = flow.nn.AdaptiveAvgPool2d((2, 2))
    of_out = m(input)
    print(of_out.numpy())


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAdaptiveAvgPool2d(flow.unittest.TestCase):
    def test_adaptive_avgpool2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_adaptive_avgpool2d_forward,
        ]
        arg_dict["device"] = ["cpu",]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
