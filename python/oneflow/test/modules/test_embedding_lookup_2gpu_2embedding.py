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
import os

placement = flow.placement("cuda", {0: [0, 1]})
batch_size = 65536

os.environ["L1_CACHE_MEMORY_BUDGET_MB"] = "4096"
os.environ["KEY_VALUE_STORE"] = "block_based"
os.environ["BLOCK_BASED_PATH"] = "/NVME0/guoran/unittest/test"


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
        options1 = {
            "embedding_name": "my_embedding1",
            "block_based_path": "/NVME0/guoran/rocks0/",
            "dtype": flow.float,
        }
        self.embedding_lookup1 = flow.nn.OneEmbeddingLookup(options1)
        options2 = {
            "embedding_name": "my_embedding2",
            "block_based_path": "/NVME1/guoran/rocks0/",
            "dtype": flow.float,
        }
        self.embedding_lookup2 = flow.nn.OneEmbeddingLookup(options2)
        self.dense1 = MatMul(3328, 1)
        self.add_optimizer(
            flow.optim.SGD(self.dense1.parameters(), lr=0.1, momentum=0.9)
        )
        self.config.enable_amp(True)
        self.grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )
        self.set_grad_scaler(self.grad_scaler)

    def build(self,):
        ids = self.data_loader()
        embedding1 = self.embedding_lookup1(ids)
        embedding2 = self.embedding_lookup2(ids)
        embedding = embedding1 + embedding2
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
