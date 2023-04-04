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
import oneflow.unittest
import oneflow.nn
import oneflow as flow
import numpy as np


class TestGridSample(flow.unittest.TestCase):
    def test_dimention_error_msg(test_case):
        N = 3
        C = 4
        H_in = 5
        H_out = 9
        inputval = oneflow.ones(N, C, H_in,)
        grid = oneflow.ones(N, H_out, 1)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.grid_sample(
                inputval, grid, mode="bilinear", padding_mode="zeros"
            )
        test_case.assertTrue("MUST be 4D or 5D input" in str(ctx.exception))

    def test_4d_gird_shape_error_msg(test_case):
        N = 3
        C = 4
        H_in = 5
        W_in = 7
        H_out = 9
        W_out = 11
        inputval = oneflow.ones(N, C, H_in, W_in)
        grid = oneflow.ones(N, H_out, W_out, 1)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.grid_sample(
                inputval, grid, mode="bilinear", padding_mode="zeros"
            )
        test_case.assertTrue(
            "Grid shape MUST (N, H_out, W_out, 2)" in str(ctx.exception)
        )

    def test_4d_grid_input_not_same_shape_error_msg(test_case):
        N = 3
        C = 4
        H_in = 5
        W_in = 7
        H_out = 9
        W_out = 11
        inputval = oneflow.ones(N, C, H_in, W_in)
        grid = oneflow.ones(N, H_out, W_out)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.grid_sample(
                inputval, grid, mode="bilinear", padding_mode="zeros"
            )
        test_case.assertTrue(
            "Grid and input MUST have same dimention" in str(ctx.exception)
        )

    def test_5d_gird_shape_error_msg(test_case):
        N = 3
        C = 4
        D_in = 8
        H_in = 5
        W_in = 7
        D_out = 13
        H_out = 9
        W_out = 11
        inputval = oneflow.ones(N, C, D_in, H_in, W_in)
        grid = oneflow.ones(N, D_out, H_out, W_out, 2)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.grid_sample(
                inputval, grid, mode="bilinear", padding_mode="zeros"
            )
        test_case.assertTrue(
            "Grid shape MUST (N, H_out, W_out, 3)" in str(ctx.exception)
        )

    def test_5d_grid_input_not_same_shape_error_msg(test_case):
        N = 3
        C = 4
        D_in = 8
        H_in = 5
        W_in = 7
        D_out = 13
        H_out = 9
        W_out = 11
        inputval = oneflow.ones(N, C, D_in, H_in, W_in)
        grid = oneflow.ones(N, D_out, H_out, W_out)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.grid_sample(
                inputval, grid, mode="bilinear", padding_mode="zeros"
            )
        test_case.assertTrue(
            "Grid and input MUST have same dimention" in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
