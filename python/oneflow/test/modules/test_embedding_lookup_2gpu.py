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
        slots = flow.randint(
            0,
            26,
            (batch_size, 26),
            placement=placement,
            sbp=flow.sbp.broadcast,
            dtype=flow.int32,
        )
        split_ids = ids.to_consistent(placement, flow.sbp.split(0))
        split_slots = slots.to_consistent(placement, flow.sbp.split(0))
        return split_ids, split_slots


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
        column_size_array = [
            227605432,
            39060,
            17295,
            7424,
            20265,
            3,
            7122,
            1543,
            63,
            130229467,
            3067956,
            405282,
            10,
            2209,
            11938,
            155,
            4,
            976,
            14,
            292775614,
            40790948,
            187188510,
            590152,
            12973,
            108,
            36,
        ]
        scales = np.sqrt(1 / np.array(column_size_array))
        initializer_list = []
        for i in range(scales.size):
            initializer_list.append(
                {
                    "initializer": {
                        "type": "uniform",
                        "low": -scales[i],
                        "high": scales[i],
                    }
                }
            )
        options = {
            "dtype": flow.float,
            "name": "my_embedding",
            "embedding_dim": 128,
            "cache": [
                {
                    "policy": "lru",
                    "cache_memory_budget_mb": 16384,
                    "value_memory_kind": "device",
                }
            ],
            "kv_store": {
                "persistent_table": {"path": "test", "physical_block_size": 512,},
            },
            "default_initializer": {"type": "normal", "mean": 0, "std": 1},
            "columns": initializer_list,
            "optimizer": {
                "lr": {
                    "base_lr": 24,
                    "decay": {
                        "type": "polynomial",
                        "decay_batches": 27772,
                        "end_lr": 0.0,
                        "power": 2.0,
                        "cycle": False,
                    },
                    "warmup": {
                        "type": "linear",
                        "warmup_batches": 2750,
                        "start_multiplier": 0.0,
                    },
                },
                "type": "sgd",
                "momentum": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        }
        self.embedding_lookup = flow.nn.OneEmbeddingLookup(options)
        self.dense1 = MatMul(3328, 1)
        self.add_optimizer(
            flow.optim.SGD(self.dense1.parameters(), lr=0.1, momentum=0.9)
        )
        self.add_optimizer(
            flow.optim.SGD(self.embedding_lookup.parameters(), lr=0.1, momentum=0.9)
        )

    def build(self,):
        ids, slots = self.data_loader()
        embedding = self.embedding_lookup(ids, slots)
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
