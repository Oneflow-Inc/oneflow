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
# export DEBUG_SHUFFLE=1
# python3 -m oneflow.distributed.launch --nproc_per_node=2 test_embedding_lookup_2gpu_unittest.py
# python3 test_unittest.py

import oneflow as flow
import numpy as np
import oneflow.nn as nn

placement = flow.placement("cuda", {0: [0, 1]})


class Model(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = flow.nn.Parameter(
            flow.randn(1, 1, 1, placement=placement, sbp=flow.sbp.broadcast)
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
    0, 1000, (10, 10), placement=placement, sbp=flow.sbp.split(0), dtype=flow.int64
)
print(ids)
graph = TrainGraph()
for i in range(1):
    loss = graph(ids)
print(loss)
