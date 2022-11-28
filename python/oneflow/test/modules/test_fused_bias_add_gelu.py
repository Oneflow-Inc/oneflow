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
import os

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_fused_bias_add_gelu(test_case, channel, axis):
    x = np.random.randn(4, channel, 8, 10)
    bias = np.random.randn(channel)
    # fused version only support in GPU
    fused_x_tensor = flow.Tensor(x).to("cuda")
    fused_x_tensor.requires_grad = True
    fused_bias_tensor = flow.Tensor(bias).to("cuda")
    fused_bias_tensor.requires_grad = True
    fused_out = flow._C.fused_bias_add_gelu(
        fused_x_tensor, fused_bias_tensor, axis=axis
    )

    origin_x_tensor = flow.Tensor(x).to("cuda")
    origin_x_tensor.requires_grad = True
    origin_bias_tensor = flow.Tensor(bias).to("cuda")
    origin_bias_tensor.requires_grad = True
    origin_out = flow.gelu(
        flow._C.bias_add(origin_x_tensor, origin_bias_tensor, axis=axis)
    )

    total_out = fused_out.sum() + origin_out.sum()
    total_out.backward()

    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    test_case.assertTrue(
        np.allclose(
            fused_x_tensor.grad.numpy(),
            origin_x_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bias_tensor.grad.numpy(),
            origin_bias_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestFusedBiasAddGelu(flow.unittest.TestCase):
    def test_gather(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_fused_bias_add_gelu]
        arg_dict["channel"] = [2, 4, 6, 8]
        arg_dict["axis"] = [1]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
