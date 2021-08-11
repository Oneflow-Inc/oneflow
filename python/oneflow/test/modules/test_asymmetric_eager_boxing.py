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
import os
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest


class TestConsistentToConsistent(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_asymmetric_consistent_to_consistent_nto1(test_case):
        if flow.distributed.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]],
                dtype=np.float32,
            )
        elif flow.distributed.get_rank() == 1:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        elif flow.distributed.get_rank() == 2:
            np_arr = np.array(
                [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]],
                dtype=np.float32,
            )
        elif flow.distributed.get_rank() == 3:
            np_arr = np.array(
                [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.Tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", {0: range(4)})
        split_tensor = tensor.to_consistent(placement, flow.sbp.split(0))
        placement2 = flow.placement("cuda", {0: range(2)})
        x = split_tensor.to_consistent(placement2, flow.sbp.split(0))
        if flow.distributed.get_rank() == 0:
            test_case.assertTrue(
                np.array_equal(
                    x.to_local().numpy(),
                    np.array(
                        [
                            [4, 6, 5, 20],
                            [6, 8, 9, 0],
                            [3, 7, 5, 0],
                            [6, 8, 9, 0],
                            [2, 10, 10, 7],
                            [3, 9, 10, 5],
                            [4, 6, 6, 9],
                            [6, 8, 6, 4],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        elif flow.distributed.get_rank() == 1:
            test_case.assertTrue(
                np.array_equal(
                    x.to_local().numpy(),
                    np.array(
                        [
                            [9, 6, 5, 8],
                            [4, 9, 7, 0],
                            [2, 5, 7, 9],
                            [6, 8, 10, 0],
                            [9, 4, 5, 8],
                            [7, 2, 9, 5],
                            [6, 3, 9, 2],
                            [3, 7, 5, 8],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        else:
            test_case.assertTrue(x.placement == placement2)


    @flow.unittest.skip_unless_1n4d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_asymmetric_consistent_to_consistent_1ton(test_case):
        if flow.distributed.get_rank() == 0:
            np_arr = np.array(
                [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]],
                dtype=np.float32,
            )
        elif flow.distributed.get_rank() == 1:
            np_arr = np.array(
                [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
                dtype=np.float32,
            )
        elif flow.distributed.get_rank() == 2:
            np_arr = np.array(
                [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]],
                dtype=np.float32,
            )
        elif flow.distributed.get_rank() == 3:
            np_arr = np.array(
                [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]],
                dtype=np.float32,
            )
        device = flow.device("cuda")
        tensor = flow.Tensor(np_arr, device=device, dtype=flow.float32)
        placement = flow.placement("cuda", {0: range(1)})
        split_tensor = tensor.to_consistent(placement, flow.sbp.split(0))
        placement2 = flow.placement("cuda", {0: range(4)})
        x = split_tensor.to_consistent(placement2, flow.sbp.split(0))
        if flow.distributed.get_rank() == 0:
            test_case.assertTrue(
                np.array_equal(
                    x.to_local().numpy(),
                    np.array(
                        [
                            [4, 6, 5, 20],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        elif flow.distributed.get_rank() == 1:
            test_case.assertTrue(
                np.array_equal(
                    x.to_local().numpy(),
                    np.array(
                        [
                            [6, 8, 9, 0],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        elif flow.distributed.get_rank() == 2:
            test_case.assertTrue(
                np.array_equal(
                    x.to_local().numpy(),
                    np.array(
                        [
                            [3, 7, 5, 0],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        elif flow.distributed.get_rank() == 3:
            test_case.assertTrue(
                np.array_equal(
                    x.to_local().numpy(),
                    np.array(
                        [
                            [6, 8, 9, 0],
                        ],
                        dtype=np.float32,
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
