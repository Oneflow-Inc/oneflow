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
import random


class OneEmbedding(nn.Module):
    def __init__(
        self,
        test_id,
        embedding_vec_size,
        persistent_path,
        table_size_array,
        size_factor,
        padding_idx,
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
            padding_idx=padding_idx,
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
        padding_idx,
    ):
        super(TestModule, self).__init__()
        self.embedding = OneEmbedding(
            test_id,
            embedding_vec_size,
            persistent_path,
            table_size_array,
            size_factor,
            padding_idx=padding_idx,
        )

    def forward(self, inputs) -> flow.Tensor:
        embedding = self.embedding(inputs)
        return embedding


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
        embedding = self.module(features.to("cuda"))
        reduce_loss = flow.mean(embedding)
        reduce_loss.backward()
        return embedding.to("cpu")


def _test_one_embedding_padding_idx(
    test_case, batch_size, table_size_array, embedding_size, test_opt, padding_idx
):
    test_str = str([batch_size, table_size_array, embedding_size, test_opt])
    test_hash = hashlib.sha256(test_str.encode("utf-8")).hexdigest()

    def np_to_global(np):
        t = flow.from_numpy(np)
        return t.to_global(placement=flow.placement.all("cpu"), sbp=flow.sbp.split(0))

    with tempfile.TemporaryDirectory() as persistent_path:
        size_factor = 3 if test_opt == "Adam" else 1
        module = TestModule(
            test_hash,
            embedding_size,
            persistent_path,
            table_size_array,
            size_factor,
            padding_idx,
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

        padding_num = random.randint(0, batch_size - 1)
        labels = np.random.randint(2, size=(batch_size, 1)).astype(np.float32)
        padding_feature = np.full(
            (len(table_size_array)), fill_value=padding_idx
        ).astype(np.int64)

        features = np.random.randint(
            sum(table_size_array), size=(batch_size, len(table_size_array))
        )
        padding_feature_idx = np.random.randint(batch_size, size=(padding_num,))
        for i in range(padding_num):
            idx = int(padding_feature_idx[i])
            features[idx] = padding_feature

        labels = np_to_global(labels)
        features = np_to_global(features)
        embedding_val = train_graph(labels, features)
        for i in range(padding_feature_idx.size):
            idx = int(padding_feature_idx[i])
            test_case.assertTrue(
                np.array_equal(
                    embedding_val[idx].numpy(),
                    np.zeros((len(table_size_array), embedding_size), dtype=np.float32),
                )
            )

        # Infer again to check the embedding in padding_idx is not updated.
        embedding_val = train_graph(labels, features)
        for i in range(padding_feature_idx.size):
            idx = int(padding_feature_idx[i])
            test_case.assertTrue(
                np.array_equal(
                    embedding_val[idx].numpy(),
                    np.zeros((len(table_size_array), embedding_size), dtype=np.float32),
                )
            )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class OneEmbeddingWithPaddingIdxTestCase(flow.unittest.TestCase):
    def test_one_embedding_padding_idx(test_case):
        arg_dict = OrderedDict()
        arg_dict["batch_size"] = [32]
        arg_dict["table_size_array"] = [
            [32, 64, 32, 32],
        ]
        arg_dict["embedding_size"] = [12]
        arg_dict["test_opt"] = ["SGD"]
        arg_dict["padding_idx"] = [2]
        os.environ["ONEFLOW_TIMEOUT_SECONDS"] = "300"
        for kwargs in GenArgDict(arg_dict):
            _test_one_embedding_padding_idx(test_case, **kwargs)
        os.environ["ONEFLOW_TIMEOUT_SECONDS"] = "90"


if __name__ == "__main__":
    unittest.main()
