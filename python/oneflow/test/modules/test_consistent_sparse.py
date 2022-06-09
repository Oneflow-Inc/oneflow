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
from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@autotest(n=1, check_graph=False)
def _test_embedding(test_case, ndim, placement, sbp):
    emb_size = random() * 8
    emb_dim = random() * 8
    emb_shape = [emb_size, emb_dim]
    idx_shape = [random(high=4) * 8 for i in range(ndim)]

    weight = random_tensor(2, *emb_shape)
    indices = random_tensor(
        len(idx_shape), *idx_shape, low=0, high=emb_size, dtype=int
    ).to_global(placement=placement, sbp=sbp)

    embedding = torch.nn.Embedding(emb_size, emb_dim, _weight=weight).to_global(
        placement=placement, sbp=sbp
    )

    output = embedding(indices)
    return output


class TestEmbedding(flow.unittest.TestCase):
    @globaltest
    def test_embedding(test_case):
        ndim = 2
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_embedding(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
