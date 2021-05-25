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
import math
import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList

def _lgamma(x):
    ret_list = []
    for i in x.flatten(): 
        ret_list.append(math.lgamma(i))
  
    return np.array(ret_list).reshape(x.shape)

    # np_input = np.random.randn(2,3)
    # y = _lgamma(np_input)
    # print(y)

def _test_lgamma_impl(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )

    of_out = flow.lgamma(of_input)
    print('of_out=', of_out)
    # np_out = np.lgamma(np_input)
    np_out = _lgamma(np_input)
    print('np_out=', np_out)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    

    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_out, 1e-4, 1e-4))
    

@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLgamma(flow.unittest.TestCase):
    def test_lgamma(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_lgamma_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
