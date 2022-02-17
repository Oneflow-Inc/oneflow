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
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestEinsum(flow.unittest.TestCase):
    @autotest(n=20, check_graph=True)
    def test_einsum_matrix_transpose(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=random(2, 6), dim1=random(2, 6),).to(device)
        z = torch.einsum("ij->ji", x)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_eltwise_multiply(test_case):
        device = random_device()
        dim0 = random(2, 6)
        dim1 = random(2, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        z = torch.einsum("ij,ij->ij", x, y)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_get_diagonal(test_case):
        device = random_device()
        dim = random(2, 6)
        x = random_tensor(ndim=2, dim0=dim, dim1=dim,).to(device)
        z = torch.einsum("ii->i", x)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_batch_permute(test_case):
        device = random_device()
        x = random_tensor(
            ndim=5,
            dim0=random(2, 6),
            dim1=random(2, 6),
            dim2=random(2, 6),
            dim3=random(2, 6),
            dim4=random(2, 6),
        ).to(device)
        z = torch.einsum("...ij->...ji", x)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_reduce_sum(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=random(2, 6), dim1=random(2, 6),).to(device)
        z = torch.einsum("ij->", x)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_matrix_column_sum(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=random(2, 6), dim1=random(2, 6),).to(device)
        z = torch.einsum("ij->j", x)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_matrix_vector_multiply(test_case):
        device = random_device()
        dim0 = random(2, 6)
        dim1 = random(2, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=1, dim0=dim1,).to(device)
        # NOTE(Liang Depeng): the same as 'ik,k->i'
        z = torch.einsum("ik,k", x, y)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_matmul(test_case):
        device = random_device()
        dim0 = random(2, 6)
        dim1 = random(2, 6)
        dim2 = random(2, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=2, dim0=dim1, dim1=dim2,).to(device)
        # NOTE(Liang Depeng): the same as 'ik,kj->ij'
        z = torch.einsum("ik,kj", x, y)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_vector_inner_product(test_case):
        device = random_device()
        dim0 = random(2, 6)
        x = random_tensor(ndim=1, dim0=dim0,).to(device)
        y = random_tensor(ndim=1, dim0=dim0,).to(device)
        # NOTE(Liang Depeng): the same as 'i,i->'
        z = torch.einsum("i,i", x, y)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_eltwise_mul_then_reduce_sum(test_case):
        device = random_device()
        dim0 = random(2, 6)
        dim1 = random(2, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        # NOTE(Liang Depeng): the same as 'ij,ij->'
        z = torch.einsum("ij,ij", x, y)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_vector_outer_product(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=random(2, 6),).to(device)
        y = random_tensor(ndim=1, dim0=random(2, 6),).to(device)
        # NOTE(Liang Depeng): the same as 'i,j->ij'
        z = torch.einsum("i,j", x, y)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_batch_matmul(test_case):
        device = random_device()
        dim0 = random(2, 6)
        dim1 = random(2, 6)
        x = random_tensor(ndim=3, dim0=dim0, dim1=random(2, 6), dim2=dim1,).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(2, 6),).to(device)
        z = torch.einsum("ijk,ikl->ijl", x, y)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_tensor_contraction(test_case):
        device = random_device()
        dim0 = random(2, 6)
        dim1 = random(2, 6)
        x = random_tensor(
            ndim=4, dim0=random(2, 6), dim1=dim0, dim2=dim1, dim3=random(2, 6),
        ).to(device)
        y = random_tensor(
            ndim=5,
            dim0=random(2, 6),
            dim1=random(2, 6),
            dim2=dim0,
            dim3=random(2, 6),
            dim4=dim1,
        ).to(device)
        z = torch.einsum("pqrs,tuqvr->pstuv", x, y)
        return z

    @autotest(n=20, check_graph=True)
    def test_einsum_bilinear_transformation(test_case):
        device = random_device()
        dim0 = random(2, 6)
        dim1 = random(2, 6)
        dim2 = random(2, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=3, dim0=random(2, 6), dim1=dim1, dim2=dim2,).to(device)
        w = random_tensor(ndim=2, dim0=dim0, dim1=dim2,).to(device)
        z = torch.einsum("ik,jkl,il->ij", x, y, w)
        return z

    @autotest(n=20, auto_backward=False, check_graph=True)
    def test_einsum_0_size_tensor(test_case):
        device = random_device()
        x = random_tensor(ndim=3, dim0=random(2, 6), dim1=0, dim2=random(2, 6),).to(
            device
        )
        z = torch.einsum("ijk", x)
        return z


if __name__ == "__main__":
    unittest.main()
