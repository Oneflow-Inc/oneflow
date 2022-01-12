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


class OneEmbeddingLookup(Module):
    def __init__(self, options):
        super().__init__()
        self.dtype = options["dtype"]
        embedding_options = {
            "embedding_name": "EmbeddingTest",
            "fixed_table_block_size": 1536,
            "base_learning_rate": 24,
            "optimizer": "adam",
            "optimizer_conf": {
                "beta1": 0.9,
                "beta2": 0.9,
                "epsilon": 0.001,
                "amsgrad": False,
                "do_bias_correction": True,
            },
            "warmup_type": "linear",
            "warmup_conf": {"warmup_batches": 2750, "start_multiplier": 0.0},
            "learning_rate_decay_type": "polynomial",
            "learning_rate_decay_conf": {
                "decay_batches": 27772,
                "end_learning_rate": 0.0,
                "power": 2.0,
                "cycle": False,
            },
        }
        self.embedding_options = json.dumps(embedding_options)

    def forward(self, ids):
        return flow._C.embedding_lookup_placeholder(
            ids, self.dtype, self.embedding_options,
        )
