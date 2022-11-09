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
class TestModule(flow.unittest.TestCase):
    def test_chunk_0_dim_input_exception(test_case):
        # torch exception and messge:
        #
        #   RuntimeError: chunk expects at least a 1-dimensional tensor.
        #
        x = flow.tensor(3.14)
        with test_case.assertRaises(RuntimeError) as ctx:
            y = flow.chunk(x, chunks=1, dim=0)
        test_case.assertTrue(
            "chunk expects at least a 1-dimensional tensor" in str(ctx.exception)
        )

    def test_chunk_0_chunks_param_exception(test_case):
        # torch exception and messge:
        #
        #   RuntimeError: chunk expects `chunks` to be greater than 0, got: 0
        #
        x = flow.tensor([[1, 2, 3], [4, 5, 6]])
        with test_case.assertRaises(RuntimeError) as ctx:
            y = flow.chunk(x, chunks=0, dim=0)
        test_case.assertTrue(
            "chunk expects `chunks` to be greater than 0, got: " in str(ctx.exception)
        )

    def test_chunk_dim_param_exception(test_case):
        # torch exception and messge:
        #
        #   IndexError: Dimension out of range (expected to be in range of [-2, 1], but got -3)
        #
        x = flow.tensor([[1, 2, 3], [4, 5, 6]])
        with test_case.assertRaises(IndexError) as ctx:
            y = flow.chunk(x, chunks=2, dim=-3)
        test_case.assertTrue(
            "Dimension out of range (expected to be in range of [-2, 1], but got -3)"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
