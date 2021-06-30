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
# import unittest
# import numpy as np
# import oneflow.experimental as flow


# flow.enable_eager_execution()
# arr = np.array([[0.25, 0.45, 0.3], [0.55, 0.32, 0.13],[0.75, 0.15, 0.1],])
# x = flow.Tensor(arr)
# x2 = flow.ones_like(x)
# y = flow.bernoulli(x2)

# print(y)


import unittest
from collections import OrderedDict

import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList


def _test_bernoulli(test_case, shape):
    input_arr = np.ones(shape)
    x = flow.Tensor(input_arr, device=flow.device("cpu"))
    y = flow.bernoulli(x)
    test_case.assertTrue(np.allclose(y.numpy(), x.numpy()))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestBernoulli(flow.unittest.TestCase):
    def test_bernoulli(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_functions"] = [
            _test_bernoulli,
        ]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
