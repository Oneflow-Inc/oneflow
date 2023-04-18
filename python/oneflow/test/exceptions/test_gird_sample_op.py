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
from oneflow.test_utils.test_util import GenArgList
import numpy as np
from collections import OrderedDict

arg_dict = OrderedDict()
arg_dict["N"] = [3, 4, 5]
arg_dict["C"] = [4, 5, 6]
arg_dict["D_in"] = [8, 11, 13]
arg_dict["H_in"] = [5, 6, 7]
arg_dict["W_in"] = [7, 8, 9]
arg_dict["D_out"] = [13, 15, 17]
arg_dict["H_out"] = [9, 10, 11]
arg_dict["W_out"] = [11, 12, 13]


def _test_dimention_error_msg_impl(test_case, N, C, H_in, H_out):
    inputval = oneflow.ones(N, C, H_in,)
    grid = oneflow.ones(N, H_out, 1)
    with test_case.assertRaises(RuntimeError) as ctx:
        flow.nn.functional.grid_sample(
            inputval, grid, mode="bilinear", padding_mode="zeros"
        )
    test_case.assertTrue("MUST be 4D or 5D input" in str(ctx.exception))


def _test_4d_gird_shape_error_msg_impl(test_case, N, C, H_in, W_in, H_out, W_out):
    inputval = oneflow.ones(N, C, H_in, W_in)
    grid = oneflow.ones(N, H_out, W_out, 1)
    with test_case.assertRaises(RuntimeError) as ctx:
        flow.nn.functional.grid_sample(
            inputval, grid, mode="bilinear", padding_mode="zeros"
        )
    test_case.assertTrue("Grid shape MUST (N, H_out, W_out, 2)" in str(ctx.exception))


def _test_4d_grid_input_not_same_shape_error_msg_impl(
    test_case, N, C, H_in, W_in, H_out, W_out
):
    inputval = oneflow.ones(N, C, H_in, W_in)
    grid = oneflow.ones(N, H_out, W_out)
    with test_case.assertRaises(RuntimeError) as ctx:
        flow.nn.functional.grid_sample(
            inputval, grid, mode="bilinear", padding_mode="zeros"
        )
    test_case.assertTrue(
        "Grid and input MUST have same dimention" in str(ctx.exception)
    )


def _test_5d_gird_shape_error_msg_impl(
    test_case, N, C, D_in, H_in, W_in, D_out, H_out, W_out
):
    inputval = oneflow.ones(N, C, D_in, H_in, W_in)
    grid = oneflow.ones(N, D_out, H_out, W_out, 2)
    with test_case.assertRaises(RuntimeError) as ctx:
        flow.nn.functional.grid_sample(
            inputval, grid, mode="bilinear", padding_mode="zeros"
        )
    test_case.assertTrue("Grid shape MUST (N, H_out, W_out, 3)" in str(ctx.exception))


def _test_5d_grid_input_not_same_shape_error_msg_impl(
    test_case, N, C, D_in, H_in, W_in, D_out, H_out, W_out
):
    inputval = oneflow.ones(N, C, D_in, H_in, W_in)
    grid = oneflow.ones(N, D_out, H_out, W_out)
    with test_case.assertRaises(RuntimeError) as ctx:
        flow.nn.functional.grid_sample(
            inputval, grid, mode="bilinear", padding_mode="zeros"
        )
    test_case.assertTrue(
        "Grid and input MUST have same dimention" in str(ctx.exception)
    )


class TestGridSample(flow.unittest.TestCase):
    def test_dimention_error_msg(test_case):
        for arg in GenArgList(arg_dict):
            _test_dimention_error_msg_impl(test_case, arg[0], arg[1], arg[3], arg[6])

    def test_4d_gird_shape_error_msg(test_case):
        for arg in GenArgList(arg_dict):
            _test_4d_gird_shape_error_msg_impl(
                test_case, arg[0], arg[1], arg[3], arg[4], arg[6], arg[7]
            )

    def test_4d_grid_input_not_same_shape_error_msg(test_case):
        for arg in GenArgList(arg_dict):
            _test_4d_grid_input_not_same_shape_error_msg_impl(
                test_case, arg[0], arg[1], arg[3], arg[4], arg[6], arg[7]
            )

    def test_5d_gird_shape_error_msg(test_case):
        for arg in GenArgList(arg_dict):
            _test_5d_gird_shape_error_msg_impl(test_case, *arg[0:])

    def test_5d_grid_input_not_same_shape_error_msg(test_case):
        for arg in GenArgList(arg_dict):
            _test_5d_grid_input_not_same_shape_error_msg_impl(test_case, *arg[0:])


if __name__ == "__main__":
    unittest.main()
