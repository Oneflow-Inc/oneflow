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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s | FileCheck %s
# CHECK-NOT: oneflow.cast

import os
import unittest
import numpy as np
import random


import oneflow as flow
import oneflow.unittest


def _cast_fuse_gn_dynamic_quant_pass(test_case):
    affine = bool(random.randint(0, 1))
    # channels_last = bool(random.randint(0, 1))
    num_channels = 8
    inp = flow.randn(4, num_channels, 32, 32).cuda()
    # if channels_last:
    #     inp = flow.randn(4, 32, 32, num_channels).cuda()
    # else:
    #     inp = flow.randn(4, num_channels, 32, 32).cuda()
    gn = flow.nn.GroupNorm(2, num_channels, affine=affine).cuda()
    kwargs = {
        "quantization_formula": "oneflow",
        "quantization_bit": 8,
        "quantization_scheme": "affine",
    }

    dynamic_quantization = flow._oneflow_internal._C.dynamic_quantization

    def fused_gn_dynamic_quant(inp, gamma, beta, affine, num_groups):
        (
            y,
            y_scale,
            y_zero_point,
        ) = flow._oneflow_internal._C.fused_group_norm_min_max_observer(
            inp, gamma, beta, affine, num_groups, **kwargs
        )
        return (
            ((y / y_scale).round() + y_zero_point).to(flow.int8),
            y_scale,
            y_zero_point,
        )

    ref_result = fused_gn_dynamic_quant(
        inp, gn.weight, gn.bias, gn.affine, gn.num_groups
    )

    class FusedGnDynamicQuantPass(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.gn = gn
            self.dynamic_quant = dynamic_quantization

        def build(self, x):
            return self.dynamic_quant(self.gn(x), **kwargs)

    lazy_b = FusedGnDynamicQuantPass()(inp)
    test_case.assertTrue(np.allclose(ref_result[0].numpy(), lazy_b[0].numpy()))
    test_case.assertTrue(np.allclose(ref_result[1].numpy(), lazy_b[1].numpy()))
    test_case.assertTrue(np.allclose(ref_result[2].numpy(), lazy_b[2].numpy()))


@flow.unittest.skip_unless_1n1d()
class TestFusedGnDynamicQuantPass(flow.unittest.MLIRTestCase):
    def setUp(self):
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
        os.environ["ONEFLOW_MLIR_STDOUT"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_TIMING"] = "1"
        os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_IR_PRINTING"] = "1"

    def test_cast_fuse_gn_dynamic_quant_pass(test_case):
        _cast_fuse_gn_dynamic_quant_pass(test_case)


if __name__ == "__main__":
    unittest.main()
