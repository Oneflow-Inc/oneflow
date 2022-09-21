"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY nnzIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest

import oneflow as flow
import oneflow.unittest
from scipy.sparse import coo_matrix
import numpy as np


from oneflow.test_utils.automated_test_util import *

@flow.unittest.skip_unless_1n1d()
class TestSpmmCooModule(flow.unittest.TestCase):
    def test_spmm_coo(test_case):
        device = "cuda"
        nnz = 9
        a_rows = 4
        a_cols = 4
        a_coo_row = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3], dtype=np.int32)
        a_coo_col = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
        a_coo_val = np.random.rand(nnz).astype(np.float32)
        b = np.random.rand(a_cols, a_rows).astype(np.float32)

        acr = flow.tensor(a_coo_row, dtype=flow.int32, device=flow.device(device))
        acc = flow.tensor(a_coo_col, dtype=flow.int32, device=flow.device(device))
        acv = flow.tensor(a_coo_val, dtype=flow.float32, device=flow.device(device))
        bb = flow.tensor(b, dtype=flow.float32, device=flow.device(device))
        flow_y = flow._C.spmm_coo(acr, acc, acv, a_rows, a_cols, bb)
        np_y = coo_matrix((a_coo_val, (a_coo_row, a_coo_col)), shape=(a_rows, a_cols)) * b

        test_case.assertTrue(
            np.allclose(flow_y.numpy(),np_y, 1e-05, 1e-05))


if __name__ == "__main__":
    unittest.main()
