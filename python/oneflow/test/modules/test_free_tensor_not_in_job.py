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
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest
import oneflow.nn as nn


class OneEmbedding(nn.Module):
    def __init__(
        self,
        table_name,
        embedding_vec_size,
        persistent_path,
        table_size_array,
        store_type,
        cache_memory_budget_mb,
        size_factor,
    ):
        assert table_size_array is not None
        vocab_size = sum(table_size_array)

        tables = [
            flow.one_embedding.make_table(
                flow.one_embedding.make_normal_initializer(mean=0.0, std=1)
            )
            for _ in range(len(table_size_array))
        ]
        if store_type == "device_mem":
            store_options = flow.one_embedding.make_device_mem_store_options(
                persistent_path=persistent_path,
                capacity=vocab_size,
                size_factor=size_factor,
            )
        elif store_type == "cached_host_mem":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_host_mem_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=vocab_size,
                size_factor=size_factor,
            )
        elif store_type == "cached_ssd":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_ssd_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=vocab_size,
                size_factor=size_factor,
            )
        else:
            raise NotImplementedError("not support", store_type)

        super(OneEmbedding, self).__init__()
        self.one_embedding = flow.one_embedding.MultiTableEmbedding(
            name=table_name,
            embedding_dim=embedding_vec_size,
            dtype=flow.float,
            key_type=flow.int64,
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)


class my_model(nn.Module):
    def __init__(
        self,
        embedding_vec_size,
        persistent_path,
        table_size_array,
        cache_memory_budget_mb,
        one_embedding_store_type,
        size_factor,
    ):
        super(my_model, self).__init__()

        self.embedding_layer = OneEmbedding(
            table_name="sparse_embedding",
            embedding_vec_size=embedding_vec_size,
            persistent_path=persistent_path,
            table_size_array=table_size_array,
            store_type=one_embedding_store_type,
            cache_memory_budget_mb=cache_memory_budget_mb,
            size_factor=size_factor,
        )

        self.linear_1 = nn.Linear(6, 6)
        self.batch_norm = nn.BatchNorm1d(6)
        self.linear_2 = nn.Linear(6, 1)

    def forward(self, X):
        embed = self.embedding_layer(X)

        y = self.linear_1(embed.flatten(start_dim=1))
        y = self.batch_norm(y)
        y = self.linear_2(y)
        return y, embed


model = my_model(
    embedding_vec_size=2,
    persistent_path="./persistent",
    table_size_array=[1, 2, 3],
    cache_memory_budget_mb=1024,
    one_embedding_store_type="cached_host_mem",
    size_factor=3,
)

model.eval()
model.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)


class Testgraph(flow.nn.Graph):
    def __init__(self, model, amp=False):
        super(Testgraph, self).__init__()
        self.module = model
        if amp:
            self.config.enable_amp(True)

    def build(self, features):
        predicts, embed = self.module(features.to("cuda"))
        return predicts.to("cpu"), embed.to("cpu")


test_graph = Testgraph(model)


@flow.unittest.skip_unless_1n1d()
class TestFreeTensorNotInJob(flow.unittest.TestCase):
    def test_free_tensor_not_in_job(test_case):
        array = np.array(
            [[0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3], [0, 2, 4], [0, 2, 5]]
        )

        X = flow.tensor(array, dtype=flow.int64)
        for i in range(6):
            x = (
                X[i]
                .reshape(1, -1)
                .to_global(
                    placement=flow.env.all_device_placement("cpu"),
                    sbp=flow.sbp.split(0),
                )
            )
            lazy_y, lazy_embed = test_graph(x)
            test_case.assertEqual(lazy_y.size(), (1, 1))
            test_case.assertEqual(lazy_embed.size(), (1, 3, 2))


if __name__ == "__main__":
    unittest.main()
