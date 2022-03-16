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

path = tempfile.TemporaryDirectory(dir="./").name


def _test_one_embedding(test_case, has_column_id, num_columns):
    batch_size = 4
    embedding_size = 2
    ids = np.random.randint(0, 1000, (batch_size, num_columns), dtype=np.int64)
    if has_column_id:
        column_ids = (
            ids % num_columns
        )  # same id must have same column id, so in this case get column_ids from ids
        column_ids_tensor = flow.tensor(
            column_ids.astype(np.int32), requires_grad=False
        ).to("cuda")
    else:
        column_ids_tensor = None
    ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")

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

    class OneEmbedding(nn.Module):
        def __init__(self):
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
                "my_embedding",
                embedding_size,
                flow.float,
                flow.int64,
                columns=initializer_list,
                store_options=store_options,
            )
            self.embedding.to("cuda")

        def forward(self, ids, column_ids):
            return self.embedding.forward(ids, column_ids)

    class TrainGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()
            self.embedding_lookup = OneEmbedding()
            self.dense = simp_module
            self.add_optimizer(
                flow.optim.SGD(self.dense.parameters(), lr=0.1, momentum=0.9)
            )
            self.add_optimizer(
                flow.optim.SGD(self.embedding_lookup.parameters(), lr=0.1, momentum=0.0)
            )

        def build(self, ids, column_ids):
            loss = self.embedding_lookup(ids, column_ids)
            loss = self.dense(loss)
            loss = loss.sum()
            loss.backward()
            return loss

    graph = TrainGraph()
    loss = graph(ids_tensor, column_ids_tensor)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class OneEmbeddingTestCase(flow.unittest.TestCase):
    def test_one_embedding(test_case):
        arg_dict = OrderedDict()
        arg_dict["has_column_id"] = [True, False]
        arg_dict["num_columns"] = [1, 26]
        for kwargs in GenArgDict(arg_dict):
            _test_one_embedding(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
