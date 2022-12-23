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


def _test_fused_scale_mask_softmax_dropout(
    test_case,
    batch_size,
    num_heads,
    seq_length,
    fill_value,
    scale_value,
    broadcast_dim,
    p,
):
    x = np.random.randn(batch_size, num_heads, seq_length, seq_length)
    mask_size = [batch_size, num_heads, seq_length, seq_length]
    if broadcast_dim:
        mask_size[broadcast_dim] = 1
    mask = np.random.randint(0, 2, size=mask_size, dtype=bool)

    fused_x_tensor = flow.tensor(x, dtype=flow.float32).to("cuda")
    fused_mask_tensor = flow.tensor(mask, dtype=flow.bool).to("cuda")
    fused_x_tensor.requires_grad = True

    # if mask is zero, fill it
    fused_out = flow._C.fused_scale_mask_softmax_dropout(
        fused_x_tensor,
        fused_mask_tensor,
        fill_value=fill_value,
        scale=scale_value,
        p=p,
    )[0]

    origin_x_tensor = flow.tensor(x, dtype=flow.float32).to("cuda")
    origin_mask_tensor = flow.tensor(mask, dtype=flow.float32).to("cuda")
    origin_x_tensor.requires_grad = True
    origin_out = flow.mul(
        origin_x_tensor, origin_mask_tensor
    ) * scale_value + fill_value * (1.0 - origin_mask_tensor)
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
class TestFusedScaleMaskSoftmaxDropout(flow.unittest.TestCase):
    def test_fused_op(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_fused_scale_mask_softmax_dropout]
        args_dict["batch_size"] = [4, 8, 16]
        args_dict["num_heads"] = [1, 4, 8]
        args_dict["seq_length"] = [8, 16, 32, 64]
        args_dict["fill_value"] = [-10000.0]
        args_dict["scale_value"] = [1.0, 2.0, 4.0]
        args_dict["broadcast_dim"] = [None, 0, 1, 2]
        args_dict["p"] = [0.0, 1.0]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
