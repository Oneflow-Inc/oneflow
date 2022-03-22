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


def _test_fused_bias_add_dropout(test_case, shape, axis, drop_prob, placement, sbp):
    x = random_tensor(len(shape), *shape).oneflow
    bias = random_tensor(1, shape[axis]).oneflow

    fused_x_tensor = x.to_global(placement=placement, sbp=sbp)
    fused_x_tensor.retain_grad()
    fused_bias_tensor = bias.to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=1).value()
    )
    fused_bias_tensor.retain_grad()
    fused_out = flow._C.fused_bias_add_dropout(
        fused_x_tensor, fused_bias_tensor, p=drop_prob, axis=axis
    )

    device = placement.type
    origin_x_tensor = x.to_local().to(device)
    origin_x_tensor.retain_grad()
    origin_bias_tensor = bias.to_local().to(device)
    origin_bias_tensor.retain_grad()

    origin_dropout = flow.nn.Dropout(p=drop_prob)
    origin_out = origin_dropout(
        flow._C.bias_add(origin_x_tensor, origin_bias_tensor, axis=axis)
    )

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
    test_case.assertTrue(
        np.allclose(
            fused_bias_tensor.grad.numpy(),
            origin_bias_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )


class TestFusedBiasAddDropout(flow.unittest.TestCase):
    @globaltest
    def test_fuse_bias_add_dropout(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(16, 64, 72), (32, 16, 48)]
        arg_dict["axis"] = [0, 1, 2, -1, -2, -3]
        arg_dict["drop_prob"] = [0.0, 1.0]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                if placement.type != "cuda":
                    continue
                for sbp in all_sbp(placement, max_dim=2):
                    _test_fused_bias_add_dropout(test_case, *arg, placement, sbp)


if __name__ == "__main__":
    unittest.main()
