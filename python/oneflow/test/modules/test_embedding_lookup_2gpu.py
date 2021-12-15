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

#python3 -m oneflow.distributed.launch --nproc_per_node=2 test_embedding_lookup_2gpu.py

import oneflow as flow
import numpy as np
import oneflow.nn as nn

np_ids = np.random.rand(10, 10).astype(np.int64)
np_embeddings = np.random.rand(10, 10, 64).astype(np.int64)
placement = flow.placement("cuda", {0: [0, 1]})


class Model(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = flow.nn.Parameter(
            flow.randn(10, 10, 64, placement=placement, sbp=flow.sbp.split(0))
        )

    def forward(self, x):
        out = x + self.w1
        return out


simp_module = Model()
simp_module.to("cuda")
simp_module.train()


class TrainGraph(flow.nn.Graph):
    def __init__(self,):
        super().__init__()
        options = {
            "name": "my_embedding",
            "embedding_size": 64,
            "dtype": flow.float,
            "encoder": "invalid",
            "partitioning": "invalid",
            "initializer": "invalid",
            "optimizer": "invalid",
            "backend": "invalid",
        }
        self.embedding_lookup = flow.nn.OneEmbeddingLookup(options)
        self.dense = simp_module
        self.add_optimizer(
            flow.optim.SGD(self.dense.parameters(), lr=0.1, momentum=0.9)
        )

    def build(self, ids):
        loss = self.embedding_lookup(ids)
        print(loss.shape)
        loss = self.dense(loss)
        loss = loss.mean()
        loss.backward()
        return loss


ids = flow.randint(
    0, 100, (10, 10), placement=placement, sbp=flow.sbp.split(0), dtype=flow.int32
)
print(ids.shape)
graph = TrainGraph()
loss = graph(ids)
print(loss)
