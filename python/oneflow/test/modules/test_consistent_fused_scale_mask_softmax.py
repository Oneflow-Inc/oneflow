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

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


def _test_fused_scale_mask_softmax(
    test_case,
    batch_size,
    num_heads,
    seq_length,
    fill_value,
    scale_value,
    placement,
    sbp,
):
    x = random_tensor(4, batch_size, num_heads, seq_length, seq_length).oneflow
    mask = random_tensor(
        4, batch_size, num_heads, seq_length, seq_length, low=0, high=2, dtype=int
    ).oneflow.to(flow.bool)

    fused_x_tensor = x.to_global(placement=placement, sbp=sbp)
    fused_mask_tensor = mask.to_global(placement=placement, sbp=sbp)
    fused_x_tensor.retain_grad()

    fused_out = flow._C.fused_scale_mask_softmax(
        fused_x_tensor, fused_mask_tensor, fill_value=fill_value, scale=scale_value,
    )

    device = placement.type
    origin_x_tensor = x.to_local().to(device)
    origin_x_tensor.retain_grad()
    origin_mask_tensor = mask.to_local().to(device, dtype=flow.float32)
    origin_out = flow.mul(
        origin_x_tensor, origin_mask_tensor
    ) * scale_value + fill_value * (1.0 - origin_mask_tensor)
    origin_out = flow.softmax(origin_out, dim=-1)

    fused_out.sum().backward()
    origin_out.sum().backward()

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


class TestFusedScaleMaskSoftmax(flow.unittest.TestCase):
    @globaltest
    def test_fused_op(test_case):
        args_dict = OrderedDict()
        args_dict["batch_size"] = [8, 16]
        args_dict["num_heads"] = [8]
        args_dict["seq_length"] = [16, 32, 64]
        args_dict["fill_value"] = [-10000.0]
        args_dict["scale_value"] = [1.0, 2.0, 4.0]

        for arg in GenArgList(args_dict):
            for placement in all_placement():
                if placement.type != "cuda":
                    continue
                for sbp in all_sbp(placement, max_dim=2):
                    _test_fused_scale_mask_softmax(test_case, *arg, placement, sbp)


if __name__ == "__main__":
    unittest.main()
