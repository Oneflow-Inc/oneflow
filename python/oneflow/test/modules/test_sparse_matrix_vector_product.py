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
import time
import datetime
import numpy as np
from collections import OrderedDict
import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng

# result compare
def compare_result(test_case, a, b, rtol=1e-5, atol=1e-8):
    test_case.assertTrue(
        np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol),
        f"\na\n{a.numpy()}\n{'-' * 80}\nb:\n{b.numpy()}\n{'*' * 80}\ndiff:\n{a.numpy() - b.numpy()}",
    )


# random vector generator
class Vector:
    def __init__(self) -> None:
        pass

    def generate_random_vector(self, m: int, dtype=flow.float32):
        self.vec = np.random.randn(m)
        self.tensor_vec = flow.FloatTensor(self.vec).to(dtype=dtype, device="cuda")


# random sparse matrix generator
class SparseMatrix:
    def __init__(self) -> None:
        self.rng = default_rng()
        self.rvs = stats.poisson(25, loc=10).rvs
        self.dtype_map = {
            flow.float32: np.float32,
            flow.float64: np.float64,
        }

    def generate_random_sparse_matrix(
        self,
        m: int,
        n: int,
        density: float,
        format: str,
        is_binary=False,
        dtype=flow.float32,
    ) -> None:
        # generate random matrix
        sparse_matrix = random(
            m=m,
            n=n,
            dtype=self.dtype_map[dtype],
            density=density,
            format=format,
            random_state=self.rng,
            data_rvs=self.rvs,
        )
        if is_binary:
            sparse_matrix.data[:] = self.dtype_map[dtype](1)

        # store random matrix
        if format == "coo":
            self.row = sparse_matrix.row
            self.col = sparse_matrix.col
        elif format == "csr":
            self.row = sparse_matrix.indptr
            self.col = sparse_matrix.indices
        elif format == "csc":
            self.row = sparse_matrix.indices
            self.col = sparse_matrix.indptr
        self.data = sparse_matrix.data
        self.scipy_matrix = sparse_matrix

        # generate tensor and store
        self.tensor_row = flow.LongTensor(self.row).to(device="cuda")
        self.tensor_col = flow.LongTensor(self.col).to(device="cuda")
        self.tensor_data = flow.FloatTensor(self.data).to(dtype=dtype, device="cuda")

        # store attributes
        self.format = format
        self.dtype = dtype
        self.num_rows = self.scipy_matrix.toarray().shape[0]

    def get_spmv_result(self, v: Vector) -> flow.Tensor:
        result_vec = np.ndarray(self.num_rows)
        if self.format == "csr":
            for i in range(self.num_rows):
                row_nnz = self.row[i + 1] - self.row[i]
                row_result = self.dtype_map[self.dtype](0)
                for j in range(row_nnz):
                    col_index = self.row[i] + j
                    row_result += self.data[col_index] * v.vec[self.col[col_index]]
                result_vec[i] = row_result
        else:
            pass

        return flow.FloatTensor(result_vec).to(dtype=self.dtype)


def _test_spmv(
    test_case, shape: dict, format: str, nnz_rate: float, dtype=flow.float32
):
    # generate random sparse matrix
    sm = SparseMatrix()
    sm.generate_random_sparse_matrix(
        m=shape["m"],
        n=shape["n"],
        density=nnz_rate,
        format=format,
        is_binary=True,
        dtype=dtype,
    )
    print(f"format: {format}, nnz_rate: {nnz_rate}, num_rows: {sm.num_rows}")

    # generate random vector
    v = Vector()
    v.generate_random_vector(m=shape["n"], dtype=dtype)

    # test gpu result
    out_vec = (
        flow._C.sparse_matrix_vector_product(
            mat_rows=sm.tensor_row,
            mat_cols=sm.tensor_col,
            mat_values=sm.tensor_data,
            in_vec=v.tensor_vec,
            format=format,
            num_rows=sm.num_rows,
        )
        .detach()
        .cpu()
    )

    # get local result
    true_vec = sm.get_spmv_result(v=v)

    # compare result
    compare_result(test_case=test_case, a=out_vec, b=true_vec, rtol=1e-5, atol=1)
    print("passed")


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestSparseMatrixVectorProduct(flow.unittest.TestCase):
    def test_gather(test_case):
        arg_dict = OrderedDict()
        # set up test functions
        arg_dict["test_fun"] = [_test_spmv]

        # set up tested matrix shape
        arg_dict["shape"] = [{"m": 512, "n": 1024}]

        # set up test sparse format
        arg_dict["format"] = ["csr"]

        # setup nnz rate
        arg_dict["nnz_rate"] = [0.005, 0.01, 0.05, 0.1]

        # setup tested data type
        arg_dict["dtype"] = [flow.float32, flow.float64]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
