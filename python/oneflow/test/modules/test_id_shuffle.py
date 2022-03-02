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
from test_util import GenArgDict
import numpy as np
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *
from test_util import GenArgList


def _test_id_shuffle(test_case, has_column_id, num_columns):
    batch_size = 55296
    ids = np.random.randint(0, 200000, (batch_size, num_columns), dtype=np.int64)
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

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, column_ids):
            return flow._C.id_shuffle(ids, column_ids, num_columns)

    graph = TestGraph()
    (
        num_unique_matrix,
        inverse_unique_partion_indices,
        cur_rank_num_unique,
        cur_rank_unique_ids,
        cur_rank_unique_column_ids,
        cur_rank_inverse_indices,
    ) = graph(ids_tensor, column_ids_tensor)
    np_unique_ids, np_inverse = np.unique(ids, return_inverse=True)
    np_num_unique = np_unique_ids.size
    test_case.assertTrue(np.array_equal(np_num_unique, num_unique_matrix[0]))
    test_case.assertTrue(np.array_equal(np_num_unique, cur_rank_num_unique[0]))
    reversed_ids = cur_rank_unique_ids[cur_rank_inverse_indices][
        inverse_unique_partion_indices
    ]
    test_case.assertTrue(np.array_equal(reversed_ids.numpy(), ids))
    if has_column_id:
        reversed_column_ids = cur_rank_unique_column_ids[cur_rank_inverse_indices][
            inverse_unique_partion_indices
        ]
        test_case.assertTrue(np.array_equal(reversed_column_ids.numpy(), column_ids))
    # when has_column_id=False, we can not test column ids because in this case same ids not lead to same column id


def _test_embedding_shuffle(test_case):
    batch_size = 55296
    num_columns = 26
    ids = np.random.randint(0, 200000, (batch_size, num_columns), dtype=np.int64)
    column_ids = (
        ids % num_columns
    )  # same id must have same column id, so in this case get column_ids from ids
    data = np.random.rand(200000, 128).astype(np.float32)
    ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
    column_ids_tensor = flow.tensor(
        column_ids.astype(np.int32), requires_grad=False
    ).to("cuda")
    data_tensor = flow.tensor(data, requires_grad=False).to("cuda")

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, column_ids, data):
            (
                num_unique_matrix,
                inverse_unique_partion_indices,
                _,
                cur_rank_unique_ids,
                _,
                cur_rank_inverse_indices,
            ) = flow._C.id_shuffle(ids, column_ids, num_columns)
            unique_embeddings = flow._C.gather(data, cur_rank_unique_ids, axis=0)
            embeddings = flow._C.embedding_shuffle(
                unique_embeddings,
                num_unique_matrix,
                cur_rank_inverse_indices,
                inverse_unique_partion_indices,
            )
            return embeddings

    graph = TestGraph()
    embeddings = graph(ids_tensor, column_ids_tensor, data_tensor)
    np_embeddings = data[ids]

    test_case.assertTrue(np.array_equal(embeddings.numpy(), np_embeddings))


def _test_embedding_gradient_shuffle(test_case):
    batch_size = 2000
    num_columns = 26
    embedding_size = 128
    ids = np.random.randint(0, 2000, (batch_size, num_columns), dtype=np.int64)
    column_ids = (
        ids % num_columns
    )  # same id must have same column id, so in this case get column_ids from ids
    embedding_diff = np.random.rand(batch_size, num_columns, embedding_size).astype(
        np.float32
    )
    ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
    column_ids_tensor = flow.tensor(
        column_ids.astype(np.int32), requires_grad=False
    ).to("cuda")
    embedding_diff_tensor = flow.tensor(embedding_diff, requires_grad=False).to("cuda")

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, column_ids, embedding_diff):
            (
                num_unique_matrix,
                inverse_unique_partion_indices,
                _,
                cur_rank_unique_ids,
                _,
                cur_rank_inverse_indices,
            ) = flow._C.id_shuffle(ids_tensor, column_ids_tensor, num_columns)
            cur_rank_unique_embedding_diff = flow._C.embedding_gradient_shuffle(
                embedding_diff,
                num_unique_matrix,
                cur_rank_inverse_indices,
                inverse_unique_partion_indices,
            )
            return (
                cur_rank_unique_embedding_diff,
                cur_rank_unique_ids,
                cur_rank_inverse_indices,
                inverse_unique_partion_indices,
            )

    graph = TestGraph()
    (
        cur_rank_unique_embedding_diff,
        cur_rank_unique_ids,
        cur_rank_inverse_indices,
        inverse_unique_partion_indices,
    ) = graph(ids_tensor, column_ids_tensor, embedding_diff_tensor)
    np_unique_ids, np_inverse = np.unique(ids, return_inverse=True)
    np_num_unique = np_unique_ids.size
    np_cur_rank_unique_embedding_diff = np.zeros(
        cur_rank_unique_embedding_diff.shape
    ).reshape(-1, embedding_size)
    for k in range(np_num_unique):
        np_cur_rank_unique_embedding_diff[k, :] = sum(
            embedding_diff.reshape(-1, embedding_size)[
                np.where(ids.flatten() == np_unique_ids[k])[0]
            ]
        )
    reversed_ids = cur_rank_unique_ids[cur_rank_inverse_indices][
        inverse_unique_partion_indices
    ]
    test_case.assertTrue(np.array_equal(reversed_ids.numpy(), ids))
    test_case.assertTrue(
        np.allclose(
            cur_rank_unique_embedding_diff[cur_rank_inverse_indices][
                inverse_unique_partion_indices
            ]
            .numpy()
            .flatten(),
            np_cur_rank_unique_embedding_diff[np_inverse].flatten(),
        )
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class FusedDotFeatureInteractionTestCase(flow.unittest.TestCase):
    def test_id_shuffle(test_case):
        arg_dict = OrderedDict()
        arg_dict["has_column_id"] = [True, False]
        arg_dict["num_columns"] = [1, 26]
        for kwargs in GenArgDict(arg_dict):
            _test_id_shuffle(test_case, **kwargs)

    def test_embedding_shuffle(test_case):
        arg_dict = OrderedDict()
        for kwargs in GenArgDict(arg_dict):
            _test_embedding_shuffle(test_case, **kwargs)

    def test_embedding_gradient_shuffle(test_case):
        arg_dict = OrderedDict()
        for kwargs in GenArgDict(arg_dict):
            _test_embedding_gradient_shuffle(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
