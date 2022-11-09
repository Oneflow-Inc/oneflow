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


@flow.unittest.skip_unless_1n1d()
class TestLayerNormModule(flow.unittest.TestCase):
    def test_layernorm_exception_input_shape_not_match(test_case):
        x = flow.randn(2, 3)
        m = flow.nn.LayerNorm(2)
        with test_case.assertRaises(RuntimeError) as ctx:
            y = m(x)
        test_case.assertTrue(
            "Given normalized_shape=(2,), expected input with shape [*, 2,], but got input of size oneflow.Size([2, 3])"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
