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
import numpy as np
import oneflow as flow
import oneflow.typing as tp

from scipy.sparse import coo_matrix


def GenerateTest(test_case, a_cooRowInd, a_cooColInd, a_cooValues, a_rows, a_cols, b):
    @flow.global_function()
    def SpmmCOOJob(
        a_cooRowInd: tp.Numpy.Placeholder((9,), dtype=flow.int64),
        a_cooColInd: tp.Numpy.Placeholder((9,), dtype=flow.int64),
        a_cooValues: tp.Numpy.Placeholder((9,), dtype=flow.float32),
        b: tp.Numpy.Placeholder((4, 3), dtype=flow.float32),
    ) -> tp.Numpy:
        with flow.scope.placement("gpu", "0:0"):
            return flow.spmm_coo(
                a_cooRowInd, a_cooColInd, a_cooValues, a_rows, a_cols, b
            )

    y = SpmmCOOJob(a_cooRowInd, a_cooColInd, a_cooValues, b)
    x = (
        coo_matrix((a_cooValues, (a_cooRowInd, a_cooColInd)), shape=(a_rows, a_cols))
        * b
    )
    test_case.assertTrue(np.array_equal(y, x))


@flow.unittest.skip_unless_1n1d()
class TestSpmmCOO(flow.unittest.TestCase):
    def test_naive(test_case):
        a_cooRowInd = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3], dtype=np.int64)
        a_cooColInd = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int64)
        a_cooValues = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32
        )
        a_rows = 4
        a_cols = 4
        b = np.array(
            [[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0], [4.0, 8.0, 12.0]],
            dtype=np.float32,
        )
        GenerateTest(
            test_case, a_cooRowInd, a_cooColInd, a_cooValues, a_rows, a_cols, b
        )


if __name__ == "__main__":
    unittest.main()
