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


def _test_fused_tril_softmax_mask_scale(
    test_case, seq_length, channel, p, diagonal, tril_scale_value
):
    x = np.random.randn(4, seq_length, channel)
    # fused version only support in GPU
    fused_x_tensor = flow.Tensor(x).to("cuda")
    fused_x_tensor.requires_grad = True
    fused_out = flow._C.fused_scale_tril_softmax_mask_scale(
        fused_x_tensor, p=p, diagonal=diagonal, tril_scale_value=tril_scale_value
    )[
        0
    ]  # The second output is softmax_y

    origin_x_tensor = flow.Tensor(x).to("cuda")
    origin_x_tensor.requires_grad = True
    origin_out = flow.tril(origin_x_tensor, diagonal)
    origin_out = origin_out * tril_scale_value
    origin_out = flow.softmax(origin_out, dim=-1)
    origin_out = flow._C.dropout(origin_out, p=p)

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


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestFusedTrilSoftmaxMaskScale(flow.unittest.TestCase):
    def test_fused_tril_softmax_dropout(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_fused_tril_softmax_mask_scale]
        arg_dict["seq_length"] = [10, 20]
        arg_dict["channel"] = [20, 30]
        arg_dict["p"] = [0.0, 1.0]
        arg_dict["diagonal"] = [0, 1, 2]
        arg_dict["tril_scale_value"] = [2, 4, 10]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
