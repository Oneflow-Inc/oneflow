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


@autotest(n=1, auto_backward=False, rtol=1.0, atol=1.0)
def _test_layernorm_wrap(test_case, placement, sbp):
    batch = 8
    channel = (random(1, 16) * 8).to(int)
    height = random(1, 2).to(int)
    width = random(1, 1024).to(int)

    def get_random_norm_shape():
        begin_axis = random(1, 3).to(int).value()
        return tuple((channel.value(), height.value(), width.value())[begin_axis:])

    x = random_tensor(4, batch, channel, height, width).to_global(
        placement=placement, sbp=sbp
    )
    m = torch.nn.LayerNorm(
        normalized_shape=get_random_norm_shape(), elementwise_affine=False,
    )

    y = m(x)
    return y


# class TestLayerNormConsistent(flow.unittest.TestCase):
#     @globaltest
#     def test_instancenorm1d(test_case):
#         for placement in all_placement():
#             for sbp in all_sbp(placement, max_dim=1):
#                 _test_layernorm_wrap(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
