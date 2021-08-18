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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_fused_bias_add_gelu(test_case):
    channel = 6
    axis = 1
    x = np.random.randn(4, channel, 8, 10)
    bias = np.random.randn(channel)
    # fused version only support in GPU
    x_tensor = flow.Tensor(x).to("cuda")
    bias_tensor = flow.Tensor(bias).to("cuda")
    fused_out = flow.F.fused_bias_add_gelu(x_tensor, bias_tensor, axis=axis)
    original_out = flow.gelu(flow.F.bias_add(x_tensor, bias_tensor, axis=axis))
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), original_out.numpy(), atol=1e-4, rtol=1e-4)
    )


@flow.unittest.skip_unless_1n1d()
class TestFusedBiasAddGelu(flow.unittest.TestCase):
    def test_gather(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_fused_bias_add_gelu]
        for arg in GenArgList(arg_dict):
            arg[0](test_case)


if __name__ == "__main__":
    unittest.main()
