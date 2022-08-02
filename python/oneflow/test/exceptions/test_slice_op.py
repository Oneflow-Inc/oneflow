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
import numpy as np


class TestSlice(flow.unittest.TestCase):
    def test_slice_update_start_list_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            ref = flow.tensor([[1], [2]])
            value = flow.tensor([[1], [2]])
            start = [-1]
            stop = [1]
            step = [1]
            flow._C.slice_update(ref, value, start, stop, step)
        test_case.assertTrue(
            "The start list elements must be greater than or equal to 0, but got"
            in str(context.exception)
        )

    def test_slice_update_stop_list_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            ref = flow.tensor([[1], [2]])
            value = flow.tensor([[1], [2]])
            start = [1]
            stop = [-1]
            step = [1]
            flow._C.slice_update(ref, value, start, stop, step)
        test_case.assertTrue(
            "The stop list elements must be greater than or equal to 0"
            in str(context.exception)
        )

    def test_slice_update_step_list_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            ref = flow.tensor([[1], [2]])
            value = flow.tensor([[1], [2]])
            start = [1]
            stop = [1]
            step = [0]
            flow._C.slice_update(ref, value, start, stop, step)
        test_case.assertTrue(
            "The step list elements must be greater than 0, but got"
            in str(context.exception)
        )

    def test_slice_update_start_and_stop_compare_value_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            ref = flow.tensor([[1], [2]])
            value = flow.tensor([[1], [2]])
            start = [2]
            stop = [1]
            step = [1]
            flow._C.slice_update(ref, value, start, stop, step)
        test_case.assertTrue(
            "The element in start list must be less than or equal to the element in stop list at index"
            in str(context.exception)
        )

    def test_slice_update_turple_size_match_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            ref = flow.tensor([1, 2])
            value = flow.tensor([1, 2])
            start = [1, 2, 3]
            stop = [1, 2, 3]
            step = [1, 2, 3]
            flow._C.slice_update(ref, value, start, stop, step)
        test_case.assertTrue(
            "The size of slice tuple must be equal to the size of value tensor at dimension"
            in str(context.exception)
        )

    def test_slice_update_type_err(test_case):
        with test_case.assertRaises(TypeError) as context:
            ref = flow.tensor([1], dtype=flow.int64)
            value = flow.tensor([0.545], dtype=flow.float32)
            start = [1]
            stop = [2]
            step = [1]
            flow._C.slice_update(ref, value, start, stop, step)
        test_case.assertTrue(
            "Tensors ref and value must have same type" in str(context.exception)
        )

    def test_slice_start_list_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            ref = flow.tensor([1])
            start = [-1]
            stop = [1]
            step = [1]
            flow._C.slice(ref, start, stop, step)
        test_case.assertTrue(
            "The start list elements must be greater than or equal to 0, but got "
            in str(context.exception)
        )

    def test_slice_stop_list_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            ref = flow.tensor([1])
            start = [1]
            stop = [-1]
            step = [1]
            flow._C.slice(ref, start, stop, step)
        test_case.assertTrue(
            "The stop list elements must be greater than or equal to 0, but got "
            in str(context.exception)
        )

    def test_slice_step_list_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            ref = flow.tensor([1])
            start = [1]
            stop = [1]
            step = [-1]
            flow._C.slice(ref, start, stop, step)
        test_case.assertTrue(
            "The step list elements must be greater than 0, but got "
            in str(context.exception)
        )

    def test_slice_start_and_stop_compare_value_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            ref = flow.tensor([1])
            start = [2]
            stop = [1]
            step = [1]
            flow._C.slice(ref, start, stop, step)
        test_case.assertTrue(
            "The element in start list must be less than or equal to the element in stop list at index "
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
