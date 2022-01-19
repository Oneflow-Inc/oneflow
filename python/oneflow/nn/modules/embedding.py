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
import warnings
from typing import Optional

import oneflow as flow
import oneflow._oneflow_internal
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _check_inplace_valid
import json
import os


fixed_table_block_size = int(os.environ.get("FIXED_TABLE_BLOCK_SIZE", 4096))
optimizer = str(os.environ.get("OPTIMIZER", "sgd"))


class OneEmbeddingLookup(Module):
    def __init__(self, options):
        super().__init__()
        self.dtype = options["dtype"]
        if options.get("embedding_name") == None:
            embedding_name = "EmbeddingTest"
        else:
            embedding_name = options["embedding_name"]
        print("embedding_name", embedding_name)
        if options.get("block_based_path") == None:
            block_based_path = os.environ.get("BLOCK_BASED_PATH")
        else:
            block_based_path = options.get("block_based_path")
        print("block_based_path", block_based_path)

        embedding_options = {
            "name": embedding_name,
            "embedding_dim": int(os.environ.get("EMBEDDING_SIZE", 128)),
            "max_query_length": int(65536 * 26),
            "l1_cache": {
                "policy": str(os.environ.get("L1_CACHE_POLICY", "lru")),
                "cache_memory_budget_mb": int(
                    os.environ.get("L1_CACHE_MEMORY_BUDGET_MB", 16384)
                ),
                "device": "device",
            },
            "l2_cache": {
                "policy": str(os.environ.get("L2_CACHE_POLICY", "none")),
                "cache_memory_budget_mb": int(
                    os.environ.get("L2_CACHE_MEMORY_BUDGET_MB", 16384)
                ),
                "device": "host",
            },
            "kv_store": {
                "persistent_table": {
                    "path": block_based_path,
                    "physical_block_size": fixed_table_block_size,
                },
            },
            "default_initializer": {"type": "uniform", "mean": 0, "std": 1},
            "columns": [
                {"initializer": {"type": "uniform", "mean": 0, "std": 1,}},
                {"initializer": {"type": "uniform", "mean": 0, "std": 1,}},
            ],
            "optimizer": {
                "type": optimizer,
                "beta": 0.9,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "do_bias_correction": True,
            },
            "learning_rate_schedule": {
                "learning_rate": 24,
                "learning_rate_decay": {
                    "type": "polynomial",
                    "decay_batches": 27772,
                    "end_learning_rate": 0.0,
                    "power": 2.0,
                    "cycle": False,
                },
                "warmup": {
                    "type": "linear",
                    "warmup_batches": 2750,
                    "start_multiplier": 0.0,
                },
            },
        }
        self.embedding_options = json.dumps(embedding_options)

    def forward(self, ids, slots):
        return flow._C.embedding_lookup_placeholder(
            ids, slots, self.dtype, self.embedding_options,
        )
