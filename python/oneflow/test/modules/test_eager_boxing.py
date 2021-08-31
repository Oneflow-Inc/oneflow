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
import oneflow as flow
import os

import oneflow.unittest
from test_util import GenArgList


def _test_eager_boxing_with_non_overlapping_placement_p_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.Tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.partial_sum)
    new_placement = flow.placement(out_device, {0: [2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertTrue(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[6, 16], [9, 17], [7, 13], [12, 16],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[15, 27], [19, 5], [11, 9], [15, 4],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_b_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.Tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    new_placement = flow.placement(out_device, {0: [2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertTrue(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[4, 6], [6, 8], [3, 7], [6, 8],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[5, 20], [9, 0], [5, 0], [9, 0],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s0_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.Tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertTrue(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[4, 6], [6, 8], [3, 7], [6, 8], [2, 10], [3, 9], [4, 6], [6, 8],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [5, 0],
                        [9, 0],
                        [10, 7],
                        [10, 5],
                        [6, 9],
                        [6, 4],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s1_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.Tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertTrue(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[4, 6], [6, 8], [3, 7], [6, 8], [2, 10], [3, 9], [4, 6], [6, 8],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [5, 0],
                        [9, 0],
                        [10, 7],
                        [10, 5],
                        [6, 9],
                        [6, 4],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s1_to_s0(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.Tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(0))
    test_case.assertTrue(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4.0, 6.0, 5.0, 20.0],
                        [6.0, 8.0, 9.0, 0.0],
                        [3.0, 7.0, 5.0, 0.0],
                        [6.0, 8.0, 9.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [2.0, 10.0, 10.0, 7.0],
                        [3.0, 9.0, 10.0, 5.0],
                        [4.0, 6.0, 6.0, 9.0],
                        [6.0, 8.0, 6.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s1_to_b(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.Tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(0))
    test_case.assertTrue(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4.0, 6.0, 5.0, 20.0],
                        [6.0, 8.0, 9.0, 0.0],
                        [3.0, 7.0, 5.0, 0.0],
                        [6.0, 8.0, 9.0, 0.0],
                        [2.0, 10.0, 10.0, 7.0],
                        [3.0, 9.0, 10.0, 5.0],
                        [4.0, 6.0, 6.0, 9.0],
                        [6.0, 8.0, 6.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4.0, 6.0, 5.0, 20.0],
                        [6.0, 8.0, 9.0, 0.0],
                        [3.0, 7.0, 5.0, 0.0],
                        [6.0, 8.0, 9.0, 0.0],
                        [2.0, 10.0, 10.0, 7.0],
                        [3.0, 9.0, 10.0, 5.0],
                        [4.0, 6.0, 6.0, 9.0],
                        [6.0, 8.0, 6.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s1_to_b(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.Tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(0))
    test_case.assertTrue(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4.0, 6.0, 5.0, 20.0],
                        [6.0, 8.0, 9.0, 0.0],
                        [3.0, 7.0, 5.0, 0.0],
                        [6.0, 8.0, 9.0, 0.0],
                        [2.0, 10.0, 10.0, 7.0],
                        [3.0, 9.0, 10.0, 5.0],
                        [4.0, 6.0, 6.0, 9.0],
                        [6.0, 8.0, 6.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingWithNonOverlappingPlacement(flow.unittest.TestCase):
    def test_eager_boxing_with_non_overlapping_placement_p_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_p_to_s1(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_b_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_b_to_s1(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s0_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s0_to_s1(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s1_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s1_to_s1(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s1_to_s0(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s1_to_s0(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s1_to_b(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s1_to_s0(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s1_to_p(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s1_to_s0(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
