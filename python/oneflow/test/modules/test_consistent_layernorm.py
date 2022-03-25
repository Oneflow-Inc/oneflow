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
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=False, auto_backward=False)
def _test_layernorm_impl(test_case, placement, sbp, get_width):
    batch = 8
    channel = 8
    height = random(1, 2).to(int)
    width = get_width().to(int)

    def get_random_norm_shape():
        begin_axis = random(1, 3).to(int).value()
        return tuple((channel, height.value(), width.value())[begin_axis:])

    norm_shape = get_random_norm_shape()
    x = random_tensor(4, batch, channel, height, width).to_global(
        placement=placement, sbp=sbp
    )
    m = torch.nn.LayerNorm(
        normalized_shape=norm_shape, elementwise_affine=oneof(True, False)
    )
    weight_and_bias_sbp = random_sbp(placement, max_dim=0)
    if m.weight is not None:
        m.weight = torch.nn.Parameter(
            m.weight.to_global(placement=placement, sbp=weight_and_bias_sbp)
        )
    if m.bias is not None:
        m.bias = torch.nn.Parameter(
            m.bias.to_global(placement=placement, sbp=weight_and_bias_sbp)
        )
    y = m(x)
    return y


class TestLayerNormConsistent(flow.unittest.TestCase):
    @globaltest
    def test_layernorm(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_layernorm_impl(test_case, placement, sbp, lambda: random(1, 1024))
                _test_layernorm_impl(
                    test_case, placement, sbp, lambda: random(1024, 8192)
                )
                _test_layernorm_impl(
                    test_case, placement, sbp, lambda: random(8192, 10240)
                )


if __name__ == "__main__":
    unittest.main()
