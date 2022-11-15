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
from oneflow.test_utils.test_util import GenArgList
import math
from typing import List

import oneflow as flow


def permute_final_dims(tensor: flow.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def _fused_op(qmk, v, scale, mask, bias, mode="row", inplace=False):
    out = flow._C.fused_msa_softmax(qmk, mask, bias, scale, mode, inplace=inplace)
    out = flow.matmul(out, v)
    return out


def _general_op(qmk, v, scale, mask, bias=None, mode="row", inplace=False):
    x = qmk * scale + mask + bias if bias is not None else qmk * scale + mask
    out = flow.softmax(x, dim=-1)
    out = flow.matmul(out, v)
    return out


def _test_fused_msa_softmax(
    test_case, N=512, S=128, D=128, h=8, d=32, mode="row", inplace=False
):
    x = flow.randn(N, S, D, requires_grad=True).cuda()  # N, S, D
    w3 = [flow.randn(D, h * d, requires_grad=True).cuda() for _ in range(3)]  # D, h*d*3
    mask = flow.randn(N, S, requires_grad=False).cuda()  # N, S
    bias = None
    scale = 1 / (d ** 0.5)
    if mode in ["row", "triangular_start", "triangular_end"]:
        bias = flow.randn(1, h, S, S, requires_grad=True).cuda()  # 1, h, S, S
        bias.retain_grad()
        mask = mask[:, None, None, :]
    if mode == "col" or mode == "global_col":
        N, S = S, N
        x = x.transpose(-2, -3)  # S, N, D
        mask = mask.transpose(-1, -2)
        if mode == "col":
            mask = mask[..., None, None, :]  # S, 1, 1, N
    q, k, v = [flow.matmul(x, w) for w in w3]  # N, S, h * d
    if mode == "template":
        n_templ = 4
        x = flow.randn(S, S, 1, D, requires_grad=True).cuda()
        k = v = flow.randn(S, S, n_templ, D, requires_grad=True).cuda()  # N, S, D
        mask = flow.randn(1, 1, 1, 1, n_templ).cuda()
        q, k, v = [flow.matmul(x_, w) for x_, w in zip([x, k, v], w3)]

    q, k, v = [
        permute_final_dims(a.view(*a.shape[:-1], h, d), (0, 2, 1, 3)) for a in [q, k, v]
    ]  # N, h, S, d

    if mode == "global_col":
        w_q = flow.randn(D, h * d, requires_grad=True).cuda()  # D, h*d
        w_kv = flow.randn(D, d * 2, requires_grad=True).cuda()  # D, h*d*2
        q = flow.sum(x * mask.unsqueeze(-1), dim=-2) / (
            flow.sum(mask, dim=-1)[..., None] + 1e-9
        )  # [N, D]
        mask = mask[..., :, None, :]  # N,1,S
        q = flow.matmul(q, w_q).view(*q.shape[:-1], h, d)  # N, h, d
        k, v = flow.matmul(x, w_kv).chunk(2, dim=-1)  # N, S, d
    qmk = flow.matmul(q, k.transpose(-1, -2))

    # general op
    x.retain_grad()
    out1 = _general_op(qmk, v, scale, mask, bias, mode, inplace)
    out1.sum().backward(retain_graph=True)
    grad_x1 = x.grad
    grad_bias1 = bias.grad if bias is not None else None

    # fused op
    out2 = _fused_op(qmk, v, scale, mask, bias, mode, inplace)
    out2.sum().backward()
    grad_x2 = x.grad / 2
    grad_bias2 = bias.grad / 2 if bias is not None else None
    test_case.assertTrue(np.allclose(out1, out2, atol=5e-3, rtol=1e-5))
    test_case.assertTrue(np.allclose(grad_x1, grad_x2, atol=1e-2, rtol=1e-5))

    if bias is not None:
        test_case.assertTrue(np.allclose(grad_bias1, grad_bias2, atol=5e-4, rtol=1e-5))


@flow.unittest.skip_unless_1n1d()
class TestFusedMsaSoftmax(flow.unittest.TestCase):
    def test_fused_msa_softmax(test_case):

        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "row")
        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "col")
        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "triangular_start")
        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "triangular_end")
        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "template")
        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "global_col")

        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "row", True)
        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "col", True)
        _test_fused_msa_softmax(
            test_case, 128, 128, 64, 8, 32, "triangular_start", True
        )
        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "triangular_end", True)
        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "template", True)
        _test_fused_msa_softmax(test_case, 16, 128, 64, 8, 32, "global_col", True)


if __name__ == "__main__":
    unittest.main()
