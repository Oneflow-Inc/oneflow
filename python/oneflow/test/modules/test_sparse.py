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

from collections import OrderedDict
import unittest
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList
import oneflow as flow
import oneflow.unittest


def _test_embedding_padding_idx(test_case, device):
    indices = flow.tensor(
        [[1, 0, 4, 8], [8, 3, 0, 9]],
        dtype=flow.int,
        device=flow.device(device),
        requires_grad=False,
    )
    embedding = flow.nn.Embedding(10, 3, padding_idx=0).to(device)
    output = embedding(indices)
    test_case.assertEqual(output[0][1].sum(), 0)
    test_case.assertEqual(output[1][2].sum(), 0)

    # negative indexing check for padding_idx
    # padding_idx=-2, num_embeddings=10 ==> index 8 padded
    embedding = flow.nn.Embedding(10, 3, padding_idx=-2).to(device)
    output = embedding(indices)
    test_case.assertEqual(output[0][3].sum(), 0)
    test_case.assertEqual(output[1][0].sum(), 0)

    # out of bounds check for padding_idx
    test_case.assertRaises(
        AssertionError,
        flow.nn.Embedding,
        num_embeddings=10,
        embedding_dim=3,
        padding_idx=25,
    )
    test_case.assertRaises(
        AssertionError,
        flow.nn.Embedding,
        num_embeddings=10,
        embedding_dim=3,
        padding_idx=-25,
    )

    padding_idx = 0
    embedding = flow.nn.Embedding(10, 3, padding_idx=padding_idx).to(device)
    indices = flow.tensor(
        [[1, 0, 4, 8], [8, 3, 0, 9]],
        dtype=flow.int,
        device=flow.device(device),
        requires_grad=False,
    )
    pre = embedding.weight[padding_idx].clone()
    embedding(indices).sum().backward()
    after = (embedding.weight + embedding.weight.grad)[padding_idx]
    embedding.zero_grad()
    test_case.assertTrue(flow.equal(after, pre).all())


@flow.unittest.skip_unless_1n1d()
class TestEmbedding(flow.unittest.TestCase):
    def test_padding_idx(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_embedding_padding_idx(test_case, *arg)

    @autotest(n=5, check_graph=True)
    def test_embedding_impl(test_case):
        device = random_device()
        emb_size = random(low=2) * 16
        emb_dim = random(low=2) * 16
        emb_shape = [emb_size, emb_dim]

        idx_ndim = random(high=4).to(int).value()
        idx_shape = [random(high=4) for i in range(idx_ndim)]

        weight = random_tensor(len(emb_shape), *emb_shape).to(device)
        indices = random_tensor(
            len(idx_shape), *idx_shape, low=0, high=emb_size, dtype=int
        ).to(device)

        embedding = torch.nn.Embedding(emb_size, emb_dim, _weight=weight).to(device)
        y = embedding(indices)
        return y

    @autotest(n=5, check_graph=True)
    def test_embedding_functional(test_case):
        device = random_device()
        emb_size = random(low=2) * 16
        emb_dim = random(low=2) * 16
        emb_shape = [emb_size, emb_dim]

        idx_ndim = random(high=4).to(int).value()
        idx_shape = [random(high=4) for i in range(idx_ndim)]

        weight = random_tensor(len(emb_shape), *emb_shape).to(device)
        indices = random_tensor(
            len(idx_shape), *idx_shape, low=0, high=emb_size, dtype=int
        ).to(device)

        y = torch.nn.functional.embedding(indices, weight)
        return y

    # @autotest(n=5, check_graph=True)
    # def test_embedding_scale_grad_by_freq(test_case):
    #     device = random_device()
    #     emb_size = random(low=2) * 16
    #     emb_dim = random(low=2) * 16
    #     emb_shape = [emb_size, emb_dim]

    #     idx_ndim = random(high=4).to(int).value()
    #     idx_shape = [random(low=2, high=4) for i in range(idx_ndim)]

    #     weight = random_tensor(len(emb_shape), *emb_shape).to(device)
    #     indices = random_tensor(
    #         len(idx_shape), *idx_shape, low=0, high=emb_size, dtype=int
    #     ).to(device)

    #     embedding = torch.nn.Embedding(
    #         emb_size, emb_dim, _weight=weight, scale_grad_by_freq=True
    #     ).to(device)
    #     y = embedding(indices)

    #     return y

    # NOTE(Yao Zihang): Set check_graph=False temporarily
    # Graph mode do not support inplace op with flow.no_grad()
    # See this issue: https://github.com/Oneflow-Inc/OneTeam/issues/1382
    @autotest(n=5, check_graph="ValidatedFlase")
    def test_embedding_renorm(test_case):
        device = random_device()
        emb_size = random(low=2) * 16
        emb_dim = random(low=2) * 16
        emb_shape = [emb_size, emb_dim]

        idx_ndim = 2
        idx_shape = [random(high=4) for i in range(idx_ndim)]

        weight = random_tensor(len(emb_shape), *emb_shape).to(device)
        indices = random_tensor(
            len(idx_shape), *idx_shape, low=0, high=emb_size, dtype=int
        ).to(device)

        embedding = torch.nn.Embedding(
            emb_size, emb_dim, max_norm=1.0, _weight=weight
        ).to(device)
        y = embedding(indices)
        return y


if __name__ == "__main__":
    unittest.main()
