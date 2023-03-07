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
    @autotest(n=5)
    def test_einsum_matrix_transpose(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=random(1, 6), dim1=random(1, 6),).to(device)
        z = torch.einsum("ij->ji", x)
        return z

    @autotest(n=5)
    def test_einsum_eltwise_multiply(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        z = torch.einsum("ij,ij->ij", x, y)
        return z

    @autotest(n=5)
    def test_einsum_get_diagonal(test_case):
        device = random_device()
        dim = random(1, 6)
        x = random_tensor(ndim=2, dim0=dim, dim1=dim,).to(device)
        z = torch.einsum("ii->i", x)
        return z

    @autotest(n=5)
    def test_einsum_batch_permute(test_case):
        device = random_device()
        x = random_tensor(
            ndim=5,
            dim0=random(1, 6),
            dim1=random(1, 6),
            dim2=random(1, 6),
            dim3=random(1, 6),
            dim4=random(1, 6),
        ).to(device)
        z = torch.einsum("...ij->...ji", x)
        return z

    @autotest(n=5)
    def test_einsum_reduce_sum(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=random(1, 6), dim1=random(1, 6),).to(device)
        z = torch.einsum("ij->", x)
        return z

    @autotest(n=5)
    def test_einsum_matrix_column_sum(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=random(1, 6), dim1=random(1, 6),).to(device)
        z = torch.einsum("ij->j", x)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_matrix_vector_multiply(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=1, dim0=dim1,).to(device)
        # NOTE(Liang Depeng): the same as 'ik,k->i'
        z = torch.einsum("ik,k", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_matmul(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        dim2 = random(1, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=2, dim0=dim1, dim1=dim2,).to(device)
        # NOTE(Liang Depeng): the same as 'ik,kj->ij'
        z = torch.einsum("ik,kj", x, y)
        return z

    @autotest(n=5)
    def test_einsum_vector_inner_product(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(ndim=1, dim0=dim0,).to(device)
        y = random_tensor(ndim=1, dim0=dim0,).to(device)
        # NOTE(Liang Depeng): the same as 'i,i->'
        z = torch.einsum("i,i", x, y)
        return z

    @autotest(n=5)
    def test_einsum_eltwise_mul_then_reduce_sum(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        # NOTE(Liang Depeng): the same as 'ij,ij->'
        z = torch.einsum("ij,ij", x, y)
        return z

    @autotest(n=5)
    def test_einsum_vector_outer_product(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=random(1, 6),).to(device)
        y = random_tensor(ndim=1, dim0=random(1, 6),).to(device)
        # NOTE(Liang Depeng): the same as 'i,j->ij'
        z = torch.einsum("i,j", x, y)
        return z

    @autotest(n=5, rtol=1e-2)
    def test_einsum_batch_matmul(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=dim1,).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 6),).to(device)
        z = torch.einsum("ijk,ikl->ijl", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_tensor_contraction(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=dim0, dim2=dim1, dim3=random(1, 6),
        ).to(device)
        y = random_tensor(
            ndim=5,
            dim0=random(1, 6),
            dim1=random(1, 6),
            dim2=dim0,
            dim3=random(1, 6),
            dim4=dim1,
        ).to(device)
        z = torch.einsum("pqrs,tuqvr->pstuv", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_bilinear_transformation(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        dim2 = random(1, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=3, dim0=random(1, 6), dim1=dim1, dim2=dim2,).to(device)
        w = random_tensor(ndim=2, dim0=dim0, dim1=dim2,).to(device)
        z = torch.einsum("ik,jkl,il->ij", x, y, w)
        return z

    @autotest(n=20, auto_backward=False, check_graph=True)
    def test_einsum_0_size_tensor(test_case):
        device = random_device()
        x = random_tensor(ndim=3, dim0=random(1, 6), dim1=0, dim2=random(1, 6),).to(
            device
        )
        z = torch.einsum("ijk", x)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_tensor_contraction2(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=dim0, dim2=random(1, 6), dim3=random(1, 6),
        ).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=random(1, 6),).to(device)
        z = torch.einsum("b n h w, n d -> b d h w", x, y)
        return z

    @autotest(n=5)
    def test_einsum_eltwise_mul_sum_row(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        z = torch.einsum("n d, n d -> n", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_matmul2(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(ndim=2, dim0=random(1, 6), dim1=dim0,).to(device)
        y = random_tensor(ndim=2, dim0=random(1, 6), dim1=dim0,).to(device)
        z = torch.einsum("i d, j d -> i j", x, y)
        return z

    @autotest(n=5, rtol=1e-3)
    def test_einsum_attention(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        dim2 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=dim2,
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=dim2,
        ).to(device)
        z = torch.einsum("b h i d, b h j d -> b h i j", x, y)
        return z

    @autotest(n=5, rtol=1e-3)
    def test_einsum_batch_matmul2(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        dim2 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=dim2
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=dim2, dim3=random(1, 6)
        ).to(device)
        z = torch.einsum("b h i j, b h j d -> b h i d", x, y)
        return z

    @autotest(n=5, rtol=1e-2)
    def test_einsum_batch_matrix_vector_multiply(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        dim2 = random(1, 6)
        x = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=dim2,).to(device)
        y = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=dim2,
        ).to(device)
        z = torch.einsum("b i d, b i j d -> b i j", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_batch_matmul3(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=dim0, dim1=random(1, 6), dim2=random(1, 6), dim3=dim1,
        ).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=dim1,).to(device)
        z = torch.einsum("b x i d, b j d -> b x i j", x, y)
        return z

    @autotest(n=5, rtol=1e-2)
    def test_einsum_batch_matmul4(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=dim0, dim1=random(1, 6), dim2=random(1, 6), dim3=dim1,
        ).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 6),).to(device)
        z = torch.einsum("b x i j, b j d -> b x i d", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_alphaflod_usecase1(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=3, dim0=random(1, 6), dim1=dim0, dim2=dim1,).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 6),).to(device)
        z = torch.einsum("hij, ijc->ihc", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_alphaflod_usecase2(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 6),).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 6),).to(device)
        z = torch.einsum("rac,rab->rbc", x, y)
        return z

    @autotest(n=5, rtol=1e-2)
    def test_einsum_alphaflod_usecase3(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 6),).to(device)
        z = torch.einsum("ra,rab->rb", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_alphaflod_usecase4(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=3, dim0=random(1, 6), dim1=dim0, dim2=dim1,).to(device)
        y = random_tensor(ndim=3, dim0=random(1, 6), dim1=dim0, dim2=dim1,).to(device)
        z = torch.einsum("qhc,khc->qkh", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_alphaflod_usecase5(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(ndim=2, dim0=random(1, 6), dim1=dim0,).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=random(1, 6),).to(
            device
        )
        z = torch.einsum("nm, mrc->nrc", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_alphaflod_usecase6(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=dim1,).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=dim1,).to(device)
        z = torch.einsum("abc,adc->bdc", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_alphaflod_usecase7(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=dim0, dim2=dim1, dim3=random(1, 6),
        ).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 6),).to(device)
        z = torch.einsum("dceb,cef->dbf", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_alphaflod_usecase8(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=random(1, 6),).to(
            device
        )
        y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=random(1, 6),).to(
            device
        )
        z = torch.einsum("acb,ade->dceb", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_alphaflod_usecase9(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(ndim=3, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0,).to(
            device
        )
        y = random_tensor(ndim=2, dim0=dim0, dim1=random(1, 6),).to(device)
        z = torch.einsum("qkc,ch->hqk", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_alphaflod_usecase10(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        dim2 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=dim2,
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=dim0, dim1=dim2, dim2=dim1, dim3=random(1, 6)
        ).to(device)
        z = torch.einsum("bhqk,bkhc->bqhc", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_alphaflod_usecase11(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(ndim=3, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0,).to(
            device
        )
        y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=random(1, 6),).to(
            device
        )
        z = torch.einsum("bqa,ahc->bqhc", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_ellipsis_usecase1(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(ndim=3, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0,).to(
            device
        )
        y = random_tensor(ndim=3, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0,).to(
            device
        )
        z = torch.einsum("...lc, ...c -> ...l", x, y)
        return z

    @autotest(n=5, rtol=1e-2)
    def test_einsum_ellipsis_usecase2(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(ndim=3, dim0=random(1, 6), dim1=dim0, dim2=dim1,).to(device)
        y = random_tensor(ndim=3, dim0=random(1, 6), dim1=dim0, dim2=dim1).to(device)
        z = torch.einsum("...lc, ...lc -> ...l", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_ellipsis_usecase3(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(ndim=3, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0,).to(
            device
        )
        y = random_tensor(ndim=3, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0).to(
            device
        )
        z = torch.einsum("...id,...jd->...ij", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_ellipsis_usecase4(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=dim0, dim2=random(1, 6), dim3=dim1
        ).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 6)).to(device)
        z = torch.einsum("...klm,kmn->...kln", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_ellipsis_usecase5(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0, dim3=random(1, 6)
        ).to(device)
        y = random_tensor(ndim=3, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0).to(
            device
        )
        z = torch.einsum("...ikl, ...jk -> ...ijl", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_ellipsis_usecase6(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(ndim=3, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0).to(
            device
        )
        y = random_tensor(ndim=3, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0).to(
            device
        )
        z = torch.einsum("...l,...l->...", x, y)
        return z

    @autotest(n=5)
    def test_einsum_ellipsis_usecase7(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        dim2 = random(1, 6)
        x = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=dim2).to(device)
        y = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=dim2, dim3=random(1, 6)
        ).to(device)
        z = torch.einsum("ijk,ijk...->ij...", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_other_usecase1(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        dim2 = random(1, 6)
        x = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=dim1).to(device)
        y = random_tensor(ndim=3, dim0=random(1, 6), dim1=dim1, dim2=dim2).to(device)
        w = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=dim2).to(device)
        z = torch.einsum("bxi,oij,byj->boxy", x, y, w)
        return z

    @autotest(n=5)
    def test_einsum_other_usecase2(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=random(1, 6)
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=random(1, 6)
        ).to(device)
        z = torch.einsum("ijac,ijkp->ijakcp", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_other_usecase3(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=dim0, dim1=random(1, 6), dim2=dim1, dim3=random(1, 6)
        ).to(device)
        y = random_tensor(ndim=3, dim0=dim0, dim1=random(1, 6), dim2=dim1).to(device)
        z = torch.einsum("cdij,cbi->cdbj", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_fastfold_usecase1(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        dim2 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=dim2
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=dim2
        ).to(device)
        z = torch.einsum("bsid,bsjd->bijd", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_fastfold_usecase2(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=random(1, 6)
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=dim0, dim1=dim1, dim2=random(1, 6), dim3=random(1, 6)
        ).to(device)
        z = torch.einsum("bsid,bsje->bijde", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_openfold_usecase1(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0, dim3=random(1, 6)
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0, dim3=random(1, 6)
        ).to(device)
        z = torch.einsum("...bac,...dae->...bdce", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_openfold_usecase2(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=dim0, dim2=random(1, 6), dim3=dim1
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=dim0, dim2=random(1, 6), dim3=dim1
        ).to(device)
        z = torch.einsum("...abc,...adc->...bdc", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-4)
    def test_einsum_openfold_usecase3(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0, dim3=dim1
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=random(1, 6), dim2=dim0, dim3=dim1
        ).to(device)
        z = torch.einsum("...qhd,...khd->...hqk", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_openfold_usecase4(test_case):
        device = random_device()
        dim0 = random(1, 6)
        dim1 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=dim0, dim2=dim1, dim3=random(1, 6)
        ).to(device)
        y = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=random(1, 6), dim2=dim1, dim3=dim0
        ).to(device)
        z = torch.einsum("...vhf,...qhv->...qhf", x, y)
        return z

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_einsum_openfold_usecase5(test_case):
        device = random_device()
        dim0 = random(1, 6)
        x = random_tensor(
            ndim=4, dim0=random(1, 6), dim1=random(1, 6), dim2=random(1, 6), dim3=dim0
        ).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=random(1, 6)).to(device)
        z = torch.einsum("...ij,jk->ik", x, y)
        return z


if __name__ == "__main__":
    unittest.main()
