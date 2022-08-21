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
import oneflow.nn as nn
import numpy as np


class OneEmbedding(nn.Module):
    def __init__(
        self, embedding_vec_size, persistent_path, key_type,
    ):
        table_size_array = [10, 20]
        vocab_size = sum(table_size_array)
        assert key_type in ["int32", "int64"], "OneEmbedding key_type must be integers"

        scales = np.sqrt(1 / np.array(table_size_array))
        tables = [
            flow.one_embedding.make_table_options(
                flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
            )
            for scale in scales
        ]
        store_options = flow.one_embedding.make_device_mem_store_options(
            persistent_path=persistent_path, capacity=vocab_size
        )

        super(OneEmbedding, self).__init__()
        self.one_embedding = flow.one_embedding.MultiTableEmbedding(
            "sparse_embedding",
            embedding_dim=embedding_vec_size,
            dtype=flow.float,
            key_type=getattr(flow, key_type),
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids, table_ids):
        return self.one_embedding.forward(ids, table_ids)


embedding = OneEmbedding(128, "test", "int32",).to("cuda")
placement = flow.placement(type="cuda", ranks=[0])
sbp = flow.sbp.split(0)
ids = (
    flow.tensor(np.random.randint(1, 10, (5, 2)).astype(np.int32), requires_grad=False)
    .to("cuda")
    .to_global(placement=placement, sbp=sbp)
)
table_ids = (
    flow.tensor(np.random.randint(0, 2, (5, 2)).astype(np.int32), requires_grad=False)
    .to("cuda")
    .to_global(placement=placement, sbp=sbp)
)
embedding.to_global(placement=placement, sbp=sbp)
y = embedding(ids, table_ids)
loss = y.sum()
print(embedding.parameters())
optimizer = flow.optim.SGD(embedding.parameters(), lr=1)
optimizer = flow.one_embedding.Optimizer(
    optimizer, embeddings=[embedding.one_embedding]
)
# 判断是不是sgd、momentum、adam

# optimizer.zero_grad()
loss.backward()
optimizer.step()


class Optimizer:
    def step(self):
        for embedding in self.embeddings:
            embedding.eager_update()
        self.optimizer.step()
