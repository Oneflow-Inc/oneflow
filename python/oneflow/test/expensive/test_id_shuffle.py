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
import numpy as np
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_id_shuffle(test_case, has_table_id, num_tables):
    batch_size = 512
    ids = np.random.randint(0, 1000, (batch_size, num_tables), dtype=np.int64)
    if has_table_id:
        table_ids = (
            ids % num_tables
        )  # same id must have same table id, so in this case get table_ids from ids
        table_ids_tensor = flow.tensor(
            table_ids.astype(np.int32), requires_grad=False
        ).to("cuda")
    else:
        table_ids_tensor = None
    ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, table_ids):
            (
                num_unique_matrix,
                inverse_unique_partition_indices,
                cur_rank_num_unique,
                cur_rank_unique_ids,
                cur_rank_unique_table_ids,
                cur_rank_inverse_indices,
            ) = flow._C.one_embedding_id_shuffle(ids, table_ids, num_tables)
            return (
                flow.cast(num_unique_matrix, flow.int32),
                flow.cast(inverse_unique_partition_indices, flow.int32),
                flow.cast(cur_rank_num_unique, flow.int32),
                flow.cast(cur_rank_unique_ids, flow.int32),
                flow.cast(cur_rank_unique_table_ids, flow.int32),
                flow.cast(cur_rank_inverse_indices, flow.int32),
            )

    graph = TestGraph()
    (
        num_unique_matrix,
        inverse_unique_partition_indices,
        cur_rank_num_unique,
        cur_rank_unique_ids,
        cur_rank_unique_table_ids,
        cur_rank_inverse_indices,
    ) = graph(ids_tensor, table_ids_tensor)
    np_unique_ids, np_inverse = np.unique(ids, return_inverse=True)
    np_num_unique = np_unique_ids.size
    test_case.assertTrue(np.array_equal(np_num_unique, num_unique_matrix[0]))
    test_case.assertTrue(np.array_equal(np_num_unique, cur_rank_num_unique[0]))
    reversed_ids = cur_rank_unique_ids[cur_rank_inverse_indices][
        inverse_unique_partition_indices
    ]
    test_case.assertTrue(np.array_equal(reversed_ids.numpy(), ids))
    if has_table_id:
        reversed_table_ids = cur_rank_unique_table_ids[cur_rank_inverse_indices][
            inverse_unique_partition_indices
        ]
        test_case.assertTrue(np.array_equal(reversed_table_ids.numpy(), table_ids))
    # when has_table_id=False, we can not test table ids because in this case same ids not lead to same table id


def _test_embedding_shuffle(test_case, dtype):
    batch_size = 512
    num_tables = 26
    ids = np.random.randint(0, 1000, (batch_size, num_tables), dtype=np.int64)
    table_ids = (
        ids % num_tables
    )  # same id must have same table id, so in this case get table_ids from ids
    if dtype == flow.float16:
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    data = np.random.rand(1000, 128).astype(np_dtype)
    ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
    table_ids_tensor = flow.tensor(table_ids.astype(np.int32), requires_grad=False).to(
        "cuda"
    )
    data_tensor = flow.tensor(data, requires_grad=False).to("cuda")

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, table_ids, data):
            (
                num_unique_matrix,
                inverse_unique_partition_indices,
                _,
                cur_rank_unique_ids,
                _,
                cur_rank_inverse_indices,
            ) = flow._C.one_embedding_id_shuffle(ids, table_ids, num_tables)
            unique_embeddings = flow._C.gather(data, cur_rank_unique_ids, axis=0)
            embeddings = flow._C.one_embedding_embedding_shuffle(
                unique_embeddings,
                num_unique_matrix,
                cur_rank_inverse_indices,
                inverse_unique_partition_indices,
            )
            return embeddings

    graph = TestGraph()
    embeddings = graph(ids_tensor, table_ids_tensor, data_tensor)
    np_embeddings = data[ids]

    test_case.assertTrue(np.array_equal(embeddings.numpy(), np_embeddings))


def _test_embedding_gradient_shuffle(test_case):
    batch_size = 512
    num_tables = 26
    embedding_size = 128
    ids = np.random.randint(0, 1000, (batch_size, num_tables), dtype=np.int64)
    table_ids = (
        ids % num_tables
    )  # same id must have same table id, so in this case get table_ids from ids
    embedding_grad = np.random.rand(batch_size, num_tables, embedding_size).astype(
        np.float32
    )
    ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
    table_ids_tensor = flow.tensor(table_ids.astype(np.int32), requires_grad=False).to(
        "cuda"
    )
    embedding_grad_tensor = flow.tensor(embedding_grad, requires_grad=False).to("cuda")

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, table_ids, embedding_grad):
            (
                num_unique_matrix,
                inverse_unique_partition_indices,
                _,
                cur_rank_unique_ids,
                _,
                cur_rank_inverse_indices,
            ) = flow._C.one_embedding_id_shuffle(ids, table_ids, num_tables)
            cur_rank_unique_embedding_grad = flow._C.one_embedding_embedding_gradient_shuffle(
                embedding_grad,
                num_unique_matrix,
                cur_rank_inverse_indices,
                inverse_unique_partition_indices,
            )
            return (
                cur_rank_unique_embedding_grad,
                flow.cast(cur_rank_unique_ids, flow.int32),
                flow.cast(cur_rank_inverse_indices, flow.int32),
                flow.cast(inverse_unique_partition_indices, flow.int32),
            )

    graph = TestGraph()
    (
        cur_rank_unique_embedding_grad,
        cur_rank_unique_ids,
        cur_rank_inverse_indices,
        inverse_unique_partition_indices,
    ) = graph(ids_tensor, table_ids_tensor, embedding_grad_tensor)
    np_unique_ids, np_inverse = np.unique(ids, return_inverse=True)
    np_num_unique = np_unique_ids.size
    np_cur_rank_unique_embedding_grad = np.zeros(
        cur_rank_unique_embedding_grad.shape
    ).reshape(-1, embedding_size)
    for k in range(np_num_unique):
        np_cur_rank_unique_embedding_grad[k, :] = sum(
            embedding_grad.reshape(-1, embedding_size)[
                np.where(ids.flatten() == np_unique_ids[k])[0]
            ]
        )
    reversed_ids = cur_rank_unique_ids[cur_rank_inverse_indices][
        inverse_unique_partition_indices
    ]
    test_case.assertTrue(np.array_equal(reversed_ids.numpy(), ids))
    test_case.assertTrue(
        np.allclose(
            cur_rank_unique_embedding_grad[cur_rank_inverse_indices][
                inverse_unique_partition_indices
            ]
            .numpy()
            .flatten(),
            np_cur_rank_unique_embedding_grad[np_inverse].flatten(),
            atol=1e-4,
            rtol=1e-4,
        )
    )


def _test_unique_key_value(test_case, has_table_id, num_tables):
    batch_size = 128
    ids = np.random.randint(0, 1000, (batch_size, num_tables), dtype=np.int64)
    if has_table_id:
        table_ids = (
            ids % num_tables
        )  # same id must have same table id, so in this case get table_ids from ids
        table_ids_tensor = flow.tensor(
            table_ids.astype(np.int32), requires_grad=False
        ).to("cuda")
    else:
        table_ids_tensor = None
    ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, table_ids):
            (
                num_unique,
                unique_ids,
                unique_table_ids,
                inverse_indices,
            ) = flow._C.one_embedding_unique_key_value_pair(ids, table_ids, num_tables)
            return (
                flow.cast(num_unique, flow.int32),
                flow.cast(unique_ids, flow.int32),
                flow.cast(unique_table_ids, flow.int32),
                flow.cast(inverse_indices, flow.int32),
            )

    graph = TestGraph()
    (num_unique, unique_ids, unique_table_ids, inverse_indices,) = graph(
        ids_tensor, table_ids_tensor
    )
    np_unique_ids, np_inverse = np.unique(ids, return_inverse=True)
    np_num_unique = np_unique_ids.size
    test_case.assertTrue(np.array_equal(np_num_unique, num_unique[0]))
    reversed_ids = unique_ids[inverse_indices]
    test_case.assertTrue(np.array_equal(reversed_ids.numpy(), ids))
    if has_table_id:
        reversed_table_ids = unique_table_ids[inverse_indices]
        test_case.assertTrue(np.array_equal(reversed_table_ids.numpy(), table_ids))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class DataShuffleTestCase(flow.unittest.TestCase):
    def test_id_shuffle(test_case):
        arg_dict = OrderedDict()
        arg_dict["has_table_id"] = [True, False]
        arg_dict["num_tables"] = [1, 26]
        for kwargs in GenArgDict(arg_dict):
            _test_id_shuffle(test_case, **kwargs)

    def test_embedding_shuffle(test_case):
        arg_dict = OrderedDict()
        arg_dict["dtype"] = [flow.float32, flow.float16]
        for kwargs in GenArgDict(arg_dict):
            _test_embedding_shuffle(test_case, **kwargs)

    def test_embedding_gradient_shuffle(test_case):
        arg_dict = OrderedDict()
        for kwargs in GenArgDict(arg_dict):
            _test_embedding_gradient_shuffle(test_case, **kwargs)

    def test_unique_key_value(test_case):
        arg_dict = OrderedDict()
        arg_dict["has_table_id"] = [True, False]
        arg_dict["num_tables"] = [13, 26, 1]
        for kwargs in GenArgDict(arg_dict):
            _test_unique_key_value(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
