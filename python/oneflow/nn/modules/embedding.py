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


class OneEmbeddingLookup(Module):
    def __init__(self, options):
        super().__init__()
        self.name = options["name"]
        self.embedding_size = options["embedding_size"]
        self.dtype = options["dtype"]
        self.encoder = options["encoder"]
        self.partitioning = options["partitioning"]
        self.initializer = options["initializer"]
        self.optimizer = options["optimizer"]
        self.backend = options["backend"]

    def forward(self, ids):
        return flow._C.embedding_lookup_placeholder(
            ids,
            self.name,
            self.embedding_size,
            self.dtype,
            self.encoder,
            self.partitioning,
            self.initializer,
            self.optimizer,
            self.backend,
        )
