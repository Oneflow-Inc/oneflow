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

import os

import unittest
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict
import numpy as np
import oneflow as flow
import oneflow.nn as nn
import tempfile
import hashlib


class OneEmbedding(nn.Module):
    def __init__(
        self,
        test_id,
        embedding_vec_size,
        persistent_path,
        table_size_array,
        size_factor,
    ):
        assert table_size_array is not None
        vocab_size = sum(table_size_array)

        scales = np.sqrt(1 / np.array(table_size_array))
        tables = [
            flow.one_embedding.make_table(
                flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
            )
            for scale in scales
        ]
        store_options = flow.one_embedding.make_device_mem_store_options(
            persistent_path=persistent_path,
            capacity=vocab_size,
            size_factor=size_factor,
        )

        super(OneEmbedding, self).__init__()
        self.one_embedding = flow.one_embedding.MultiTableEmbedding(
            f"oneembedding_{test_id}",
            embedding_dim=embedding_vec_size,
            dtype=flow.float,
            key_type=flow.int64,
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)


class TestModule(nn.Module):
    def __init__(
        self,
        test_id,
        embedding_vec_size,
        persistent_path,
        table_size_array,
        size_factor,
    ):
        super(TestModule, self).__init__()
        self.embedding = OneEmbedding(
            test_id, embedding_vec_size, persistent_path, table_size_array, size_factor
        )
        self.mlp = nn.Linear(embedding_vec_size, 1)

    def forward(self, inputs) -> flow.Tensor:
        embedding = self.embedding(inputs)
        logits = self.mlp(embedding).mean(dim=1)
        return logits


class TrainGraph(flow.nn.Graph):
    def __init__(
        self, module, loss, optimizer, amp=False,
    ):
        super(TrainGraph, self).__init__()
        self.module = module
        self.loss = loss
        self.add_optimizer(optimizer)
        if amp:
            self.config.enable_amp(True)

    def build(self, labels, features):
        logits = self.module(features.to("cuda"))
        loss = self.loss(logits, labels.to("cuda"))
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss.to("cpu")


def _test_one_embedding(
    test_case, batch_size, table_size_array, embedding_size, test_opt
):
    test_str = str([batch_size, table_size_array, embedding_size, test_opt])
    test_hash = hashlib.sha256(test_str.encode("utf-8")).hexdigest()

    def np_to_global(np):
        t = flow.from_numpy(np)
        return t.to_global(placement=flow.placement.all("cpu"), sbp=flow.sbp.split(0))

    with tempfile.TemporaryDirectory() as persistent_path:
        size_factor = 3 if test_opt == "Adam" else 1
        module = TestModule(
            test_hash, embedding_size, persistent_path, table_size_array, size_factor
        )
        module.to_global(flow.placement.all("cuda"), flow.sbp.broadcast)

        if test_opt == "Adam":
            opt = flow.optim.Adam(module.parameters(), lr=0.1)
        elif test_opt == "SGD":
            opt = flow.optim.SGD(module.parameters(), lr=0.1)
        else:
            assert False

        loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")

        train_graph = TrainGraph(module, loss, opt)

        module.train()
        for step in range(1, 101):
            labels = np.random.randint(2, size=(batch_size, 1)).astype(np.float32)
            features = np.random.randint(
                sum(table_size_array), size=(batch_size, len(table_size_array))
            )
            labels = np_to_global(labels)
            features = np_to_global(features)
            loss = train_graph(labels, features)
            test_case.assertFalse(np.isnan(loss.numpy()))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class OneEmbeddingTestCase(flow.unittest.TestCase):
    def test_one_embedding(test_case):
        arg_dict = OrderedDict()
        arg_dict["batch_size"] = [32, 4096]
        arg_dict["table_size_array"] = [
            [32, 65536, 100, 7],
            [32768, 10000, 17, 3, 686],
        ]
        arg_dict["embedding_size"] = [128, 17]
        arg_dict["test_opt"] = ["SGD", "Adam"]
        for kwargs in GenArgDict(arg_dict):
            _test_one_embedding(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
