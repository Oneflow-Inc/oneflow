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

# python3 -m oneflow.distributed.launch --nproc_per_node=2 test_embedding_lookup_2gpu.py

import oneflow as flow
import numpy as np
import oneflow.nn as nn

placement = flow.placement("cuda", {0: [0, 1]})
batch_size = 65536


class SyntheticDataLoader(nn.Module):
    def __init__(self,):
        super(SyntheticDataLoader, self).__init__()
        print("use synthetic data")

    def forward(self):
        ids = flow.randint(
            0,
            150000,
            (batch_size, 26),
            placement=placement,
            sbp=flow.sbp.broadcast,
            dtype=flow.int64,
        )
        split_ids = ids.to_consistent(placement, flow.sbp.split(0))
        return split_ids


class MatMul(flow.nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.w1 = flow.nn.Parameter(
            flow.randn(k, 1024, placement=placement, sbp=flow.sbp.broadcast)
        )
        self.w2 = flow.nn.Parameter(
            flow.randn(1024, 512, placement=placement, sbp=flow.sbp.broadcast)
        )
        self.w3 = flow.nn.Parameter(
            flow.randn(512, 1, placement=placement, sbp=flow.sbp.broadcast)
        )

    def forward(self, x):
        out = flow.matmul(x, self.w1)
        out = flow.matmul(out, self.w2)
        out = flow.matmul(out, self.w3)
        return out


class TrainGraph(flow.nn.Graph):
    def __init__(self,):
        super().__init__()
        self.data_loader = SyntheticDataLoader()
        options = {
            "name": "my_embedding",
            # Can't change the embedding_size 128 because the kv store value_length has been set to 128
            "embedding_size": 128,
            "dtype": flow.float,
            "encoder": "invalid",
            "partitioning": "invalid",
            "initializer": "invalid",
            "optimizer": "invalid",
            "backend": "invalid",
        }
        # Can't change the name 'embedding_lookup' because it is used to generate backward and optimizer
        self.embedding_lookup = flow.nn.OneEmbeddingLookup(options)
        self.dense1 = MatMul(3328, 1)
        self.add_optimizer(
            flow.optim.SGD(self.dense1.parameters(), lr=0.1, momentum=0.9)
        )

    def build(self,):
        ids = self.data_loader()
        embedding = self.embedding_lookup(ids)
        loss = embedding.reshape(batch_size, -1)
        print(loss.shape)
        loss = self.dense1(loss)
        loss = loss.mean()
        loss.backward()
        return loss


graph = TrainGraph()
for i in range(20):
    loss = graph()
print(loss)
