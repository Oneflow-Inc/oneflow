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

import oneflow as flow
import numpy as np
import oneflow.nn as nn


class Model(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = flow.nn.Parameter(flow.tensor([[0]], dtype=flow.float32))

    def forward(self, x):
        out = x + self.w1
        return out


simp_module = Model()
simp_module.to("cuda")
simp_module.train()


class TrainGraph(flow.nn.Graph):
    def __init__(self,):
        super().__init__()
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
        self.dense = simp_module
        self.add_optimizer(
            flow.optim.SGD(self.dense.parameters(), lr=0.1, momentum=0.9)
        )
        # self.add_optimizer(
        #    flow.optim.SGD(self.embedding_lookup.parameters(), lr=0.1, momentum=0.9)
        # )

    def build(self, ids, column_ids):
        loss = self.embedding_lookup(ids, column_ids)
        loss = self.dense(loss)
        print(loss)
        loss = loss.sum()
        loss.backward()
        return loss


np_ids = np.random.rand(10, 26).astype(np.int64)
np_column_ids = np.random.randint(0, 26, (10, 26), dtype=np.int32)
ids = flow.tensor(np_ids).to("cuda")
column_ids = flow.tensor(np_column_ids).to("cuda")
graph = TrainGraph()
loss = graph(ids, column_ids)
print(loss)
