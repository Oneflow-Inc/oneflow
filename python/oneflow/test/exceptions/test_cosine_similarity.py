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
class TestCosineSimilarity(flow.unittest.TestCase):
    def test_cosine_similarity_not_floating_type(test_case):
        x = flow.randn(2, 5).to(flow.int32)
        y = flow.randn(2, 5).to(flow.int32)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.cosine_similarity(x, y, dim=1)
        test_case.assertTrue(
            "expected common dtype to be floating point, yet common dtype is oneflow.int32"
            in str(ctx.exception)
        )

    def test_cosine_similarity_broadcast(test_case):
        x = flow.randn(2, 5)
        y = flow.randn(2, 4)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.cosine_similarity(x, y, dim=1)
        test_case.assertTrue(
            "The size of tensor a (5) must match the size of tensor b (4) at non-singleton dimension 1"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
