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
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=2, check_graph=False)
def _test_einsum_matrix_transpose(test_case, placement, sbp):
    x = random_tensor(ndim=2, dim0=random(1, 3) * 8, dim1=random(1, 3) * 8)
    g_x = x.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ij->ji", g_x)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_eltwise_multiply(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ij,ij->ij", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_get_diagonal(test_case, placement, sbp):
    dim = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim, dim1=dim,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ii->i", g_x)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_batch_permute(test_case, placement, sbp):
    x = random_tensor(
        ndim=5,
        dim0=random(1, 3) * 8,
        dim1=random(1, 3) * 8,
        dim2=random(1, 3) * 8,
        dim3=random(1, 3) * 8,
        dim4=random(1, 3) * 8,
    )
    g_x = x.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("...ij->...ji", g_x)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_reduce_sum(test_case, placement, sbp):
    x = random_tensor(ndim=2, dim0=random(1, 3) * 8, dim1=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ij->", g_x)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_matrix_column_sum(test_case, placement, sbp):
    x = random_tensor(ndim=2, dim0=random(1, 3) * 8, dim1=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ij->j", g_x)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_matrix_vector_multiply(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=1, dim0=dim1,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    # NOTE(Liang Depeng): the same as 'ik,k->i'
    z = torch.einsum("ik,k", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_matmul(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    dim2 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=2, dim0=dim1, dim1=dim2,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    # NOTE(Liang Depeng): the same as 'ik,kj->ij'
    z = torch.einsum("ik,kj", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_vector_inner_product(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    x = random_tensor(ndim=1, dim0=dim0,)
    y = random_tensor(ndim=1, dim0=dim0,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    # NOTE(Liang Depeng): the same as 'i,i->'
    z = torch.einsum("i,i", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_eltwise_mul_then_reduce_sum(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    # NOTE(Liang Depeng): the same as 'ij,ij->'
    z = torch.einsum("ij,ij", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_vector_outer_product(test_case, placement, sbp):
    x = random_tensor(ndim=1, dim0=random(1, 3) * 8,)
    y = random_tensor(ndim=1, dim0=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    # NOTE(Liang Depeng): the same as 'i,j->ij'
    z = torch.einsum("i,j", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_batch_matmul(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 3) * 8, dim2=dim1,)
    y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ijk,ikl->ijl", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_tensor_contraction(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(
        ndim=4, dim0=random(1, 3) * 8, dim1=dim0, dim2=dim1, dim3=random(1, 3) * 8,
    )
    y = random_tensor(
        ndim=5,
        dim0=random(1, 3) * 8,
        dim1=random(1, 3) * 8,
        dim2=dim0,
        dim3=random(1, 3) * 8,
        dim4=dim1,
    )
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("pqrs,tuqvr->pstuv", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_bilinear_transformation(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    dim2 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=3, dim0=random(1, 3) * 8, dim1=dim1, dim2=dim2,)
    w = random_tensor(ndim=2, dim0=dim0, dim1=dim2,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    g_w = w.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ik,jkl,il->ij", g_x, g_y, g_w)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_tensor_contraction2(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    x = random_tensor(
        ndim=4,
        dim0=random(1, 3) * 8,
        dim1=dim0,
        dim2=random(1, 3) * 8,
        dim3=random(1, 3) * 8,
    )
    y = random_tensor(ndim=2, dim0=dim0, dim1=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("b n h w, n d -> b d h w", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_eltwise_mul_sum_row(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("n d, n d -> n", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_matmul2(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=random(1, 3) * 8, dim1=dim0,)
    y = random_tensor(ndim=2, dim0=random(1, 3) * 8, dim1=dim0,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("i d, j d -> i j", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_attention(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    dim2 = random(1, 3) * 8
    x = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8, dim3=dim2,)
    y = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8, dim3=dim2,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("b h i d, b h j d -> b h i j", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_batch_matmul2(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    dim2 = random(1, 3) * 8
    x = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8, dim3=dim2)
    y = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dim2=dim2, dim3=random(1, 3) * 8)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("b h i j, b h j d -> b h i d", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_batch_matrix_vector_multiply(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    dim2 = random(1, 3) * 8
    x = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=dim2,)
    y = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8, dim3=dim2,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("b i d, b i j d -> b i j", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_batch_matmul3(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(
        ndim=4, dim0=dim0, dim1=random(1, 3) * 8, dim2=random(1, 3) * 8, dim3=dim1,
    )
    y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 3) * 8, dim2=dim1,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("b x i d, b j d -> b x i j", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_batch_matmul4(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(
        ndim=4, dim0=dim0, dim1=random(1, 3) * 8, dim2=random(1, 3) * 8, dim3=dim1,
    )
    y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("b x i j, b j d -> b x i d", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase1(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=3, dim0=random(1, 3) * 8, dim1=dim0, dim2=dim1,)
    y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("hij, ijc->ihc", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase2(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8,)
    y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("rac,rab->rbc", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase3(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ra,rab->rb", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase4(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=3, dim0=random(1, 3) * 8, dim1=dim0, dim2=dim1,)
    y = random_tensor(ndim=3, dim0=random(1, 3) * 8, dim1=dim0, dim2=dim1,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("qhc,khc->qkh", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase5(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=random(1, 3) * 8, dim1=dim0,)
    y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 3) * 8, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("nm, mrc->nrc", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase6(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 3) * 8, dim2=dim1,)
    y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 3) * 8, dim2=dim1,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("abc,adc->bdc", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase7(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(
        ndim=4, dim0=random(1, 3) * 8, dim1=dim0, dim2=dim1, dim3=random(1, 3) * 8,
    )
    y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("dceb,cef->dbf", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase8(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    x = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 3) * 8, dim2=random(1, 3) * 8,)
    y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 3) * 8, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("acb,ade->dceb", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase9(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    x = random_tensor(ndim=3, dim0=random(1, 3) * 8, dim1=random(1, 3) * 8, dim2=dim0,)
    y = random_tensor(ndim=2, dim0=dim0, dim1=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("qkc,ch->hqk", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase10(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    dim2 = random(1, 3) * 8
    x = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8, dim3=dim2,)
    y = random_tensor(ndim=4, dim0=dim0, dim1=dim2, dim2=dim1, dim3=random(1, 3) * 8)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("bhqk,bkhc->bqhc", g_x, g_y)
    return z


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase11(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    x = random_tensor(ndim=3, dim0=random(1, 3) * 8, dim1=random(1, 3) * 8, dim2=dim0,)
    y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 3) * 8, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("bqa,ahc->bqhc", g_x, g_y)
    return z


class TestEinsumConsistent(flow.unittest.TestCase):
    @globaltest
    def test_einsum_matrix_transpose(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_matrix_transpose(test_case, placement, sbp)

    @globaltest
    def test_einsum_eltwise_multiply(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_eltwise_multiply(test_case, placement, sbp)

    @globaltest
    def test_einsum_get_diagonal(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_get_diagonal(test_case, placement, sbp)

    @globaltest
    def test_einsum_batch_permute(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=5):
                _test_einsum_batch_permute(test_case, placement, sbp)

    @globaltest
    def test_einsum_reduce_sum(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_reduce_sum(test_case, placement, sbp)

    @globaltest
    def test_einsum_matrix_column_sum(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_matrix_column_sum(test_case, placement, sbp)

    @globaltest
    def test_einsum_matrix_vector_multiply(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_einsum_matrix_vector_multiply(test_case, placement, sbp)

    @globaltest
    def test_einsum_matmul(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_matmul(test_case, placement, sbp)

    @globaltest
    def test_einsum_vector_inner_product(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_einsum_vector_inner_product(test_case, placement, sbp)

    @globaltest
    def test_einsum_eltwise_mul_then_reduce_sum(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_eltwise_mul_then_reduce_sum(test_case, placement, sbp)

    @globaltest
    def test_einsum_vector_outer_product(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_einsum_vector_outer_product(test_case, placement, sbp)

    @globaltest
    def test_einsum_batch_matmul(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_batch_matmul(test_case, placement, sbp)

    @globaltest
    def test_einsum_tensor_contraction(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_einsum_tensor_contraction(test_case, placement, sbp)

    @globaltest
    def test_einsum_bilinear_transformation(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_bilinear_transformation(test_case, placement, sbp)

    @globaltest
    def test_einsum_tensor_contraction2(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_tensor_contraction2(test_case, placement, sbp)

    @globaltest
    def test_einsum_eltwise_mul_sum_row(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_eltwise_mul_sum_row(test_case, placement, sbp)

    @globaltest
    def test_einsum_matmul2(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_matmul2(test_case, placement, sbp)

    @globaltest
    def test_einsum_attention(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_einsum_attention(test_case, placement, sbp)

    @globaltest
    def test_einsum_batch_matmul2(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_einsum_batch_matmul2(test_case, placement, sbp)

    @globaltest
    def test_einsum_batch_matrix_vector_multiply(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_batch_matrix_vector_multiply(test_case, placement, sbp)

    @globaltest
    def test_einsum_batch_matmul3(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_batch_matmul3(test_case, placement, sbp)

    @globaltest
    def test_einsum_batch_matmul4(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_batch_matmul4(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase1(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_alphaflod_usecase1(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase2(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_alphaflod_usecase2(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase3(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_alphaflod_usecase3(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase4(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_alphaflod_usecase4(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase5(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_alphaflod_usecase5(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase6(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_alphaflod_usecase6(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase7(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_alphaflod_usecase7(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase8(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_alphaflod_usecase8(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase9(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_alphaflod_usecase9(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase10(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_einsum_alphaflod_usecase10(test_case, placement, sbp)

    @globaltest
    def test_einsum_alphaflod_usecase11(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_einsum_alphaflod_usecase11(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
