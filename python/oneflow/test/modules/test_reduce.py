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
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList


def _test_reduce(test_case, dst, device):
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
    x = flow.tensor(np_arr, device=device, dtype=flow.float32)
    flow._C.local_reduce(x, dst=dst)
    if flow.env.get_rank() == dst:
        test_case.assertTrue(
            np.allclose(
                x.numpy(),
                np.array(
                    [
                        [24, 26, 25, 43],
                        [20, 28, 35, 10],
                        [15, 21, 27, 20],
                        [21, 31, 30, 12],
                    ],
                    dtype=np.float32,
                ),
            )
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n4d()
class TestReduce(flow.unittest.TestCase):
    def test_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["dst"] = [0, 1, 2, 3]
        arg_dict["device"] = ["cpu", "cuda"]

        for arg in GenArgList(arg_dict):
            _test_reduce(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
