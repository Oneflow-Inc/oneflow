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
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict
import oneflow as flow
import numpy as np
import oneflow.nn as nn
import oneflow.unittest
import tempfile

from oneflow.test_utils.automated_test_util import *

path1 = tempfile.TemporaryDirectory(dir="./").name
path2 = tempfile.TemporaryDirectory(dir="./").name


def _test_one_embedding(test_case, has_column_id, num_columns, use_fp16):
    placement = flow.placement(type="cuda", ranks=list(range(2)))
    batch_size = 4
    embedding_size = 2
    ids = np.random.randint(0, 1000, (batch_size, num_columns), dtype=np.int64)
    ids_tensor = flow.tensor(ids, requires_grad=False).to_global(
        placement=placement, sbp=flow.sbp.split(0)
    )
    if has_column_id:
        column_ids = (
            ids % num_columns
        )  # same id must have same column id, so in this case get column_ids from ids
        column_ids_tensor = flow.tensor(
            column_ids.astype(np.int32), requires_grad=False
        ).to_global(placement=placement, sbp=flow.sbp.split(0))
    else:
        column_ids_tensor = None

    class MatMul(flow.nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.w1 = flow.nn.Parameter(
                flow.randn(k, 1, placement=placement, sbp=flow.sbp.broadcast)
            )

        def forward(self, x):
            out = flow.matmul(x, self.w1)
            return out

    class OneEmbedding(nn.Module):
        def __init__(self, name, path):
            super().__init__()
            column_size_array = [np.random.randint(100, 1000)] * num_columns
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
            store_options = flow.one_embedding.make_device_mem_cached_ssd_store_options(
                device_memory_mb=16, persistent_path=path, size_factor=1,
            )
            self.embedding = flow.one_embedding.Embedding(
                name,
                embedding_size,
                flow.float,
                flow.int64,
                columns=initializer_list,
                store_options=store_options,
            )
            self.embedding = self.embedding.to_global(
                placement=placement, sbp=flow.sbp.broadcast
            )

        def forward(self, ids, column_ids):
            return self.embedding.forward(ids, column_ids)

    class TrainGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()
            if use_fp16:
                self.config.enable_amp(True)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=1073741824,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                self.set_grad_scaler(grad_scaler)
            self.dense = MatMul(embedding_size * num_columns, 1)
            self.embedding_lookup1 = OneEmbedding("emb1", path1)
            self.embedding_lookup2 = OneEmbedding("emb2", path2)
            self.add_optimizer(
                flow.optim.SGD(self.dense.parameters(), lr=0.1, momentum=0.0)
            )
            self.add_optimizer(
                flow.optim.SGD(
                    self.embedding_lookup1.parameters(), lr=0.1, momentum=0.0
                )
            )
            self.add_optimizer(
                flow.optim.SGD(
                    self.embedding_lookup2.parameters(), lr=0.1, momentum=0.0
                )
            )

        def build(self, ids, column_ids):
            embedding1 = self.embedding_lookup1.forward(ids, column_ids)
            embedding2 = self.embedding_lookup2.forward(ids, column_ids)
            embedding = embedding1 + embedding2
            loss = embedding.reshape(embedding.shape[0], -1)
            loss = self.dense(loss)
            loss = loss.mean()
            loss.backward()
            return loss

    graph = TrainGraph()
    loss = graph(ids_tensor, column_ids_tensor)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class OneEmbeddingTestCase(flow.unittest.TestCase):
    def test_one_embedding1(test_case):
        arg_dict = OrderedDict()
        arg_dict["has_column_id"] = [True, False]
        arg_dict["num_columns"] = [1, 2]
        arg_dict["use_fp16"] = [False]
        for kwargs in GenArgDict(arg_dict):
            _test_one_embedding(test_case, **kwargs)

    def test_one_embedding2(test_case):
        arg_dict = OrderedDict()
        arg_dict["has_column_id"] = [True]
        arg_dict["num_columns"] = [26]
        arg_dict["use_fp16"] = [True]
        for kwargs in GenArgDict(arg_dict):
            _test_one_embedding(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
