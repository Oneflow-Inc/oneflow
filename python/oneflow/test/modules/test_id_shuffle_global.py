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

parallel_num = 2
max_id = 1000


def get_tensors(batch_size, num_columns):
    placement = flow.placement(type="cuda", ranks=list(range(parallel_num)))
    ids = np.random.randint(0, max_id, (batch_size, num_columns), dtype=np.int64)
    ids_tensor = flow.tensor(ids, requires_grad=False).to_global(
        placement=placement, sbp=flow.sbp.split(0)
    )
    column_ids = (
        ids % num_columns
    )  # same id must have same column id, so in this case get column_ids from ids
    column_ids_tensor = flow.tensor(
        column_ids.astype(np.int32), requires_grad=False
    ).to_global(placement=placement, sbp=flow.sbp.split(0))
    return ids_tensor, column_ids_tensor


def _test_id_shuffle(test_case, has_column_id, num_columns):
    batch_size = int(1024 / parallel_num)
    placement = flow.placement(type="cuda", ranks=list(range(parallel_num)))

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, column_ids):
            (
                num_unique_matrix,
                inverse_unique_partition_indices,
                cur_rank_num_unique,
                cur_rank_unique_ids,
                cur_rank_unique_column_ids,
                cur_rank_inverse_indices,
            ) = flow._C.one_embedding_id_shuffle(ids, column_ids, num_columns)
            return (
                flow.cast(num_unique_matrix, flow.int32),
                flow.cast(inverse_unique_partition_indices, flow.int32),
                flow.cast(cur_rank_num_unique, flow.int32),
                flow.cast(cur_rank_unique_ids, flow.int32),
                flow.cast(cur_rank_unique_column_ids, flow.int32),
                flow.cast(cur_rank_inverse_indices, flow.int32),
            )

    graph = TestGraph()
    for i in range(10):
        ids_tensor, column_ids_tensor = get_tensors(batch_size, num_columns)
        if not has_column_id:
            column_ids_tensor = None
        graph(ids_tensor, column_ids_tensor)
    (
        num_unique_matrix,
        inverse_unique_partition_indices,
        local_cur_rank_num_unique,
        cur_rank_unique_ids,
        cur_rank_unique_column_ids,
        cur_rank_inverse_indices,
    ) = graph(ids_tensor, column_ids_tensor)
    cur_rank_num_unique = local_cur_rank_num_unique.to_local().to_global(
        placement=placement, sbp=flow.sbp.split(0)
    )
    cur_rank_num_unique_list = []
    cur_rank_unique_ids_list = []
    cur_rank_unique_column_ids_list = []
    cur_rank_num_ids = batch_size * num_columns * parallel_num
    for i in range(parallel_num):
        num_unique_i = cur_rank_num_unique.numpy()[i]
        unique_ids_i = cur_rank_unique_ids.numpy()[
            cur_rank_num_ids * i : cur_rank_num_ids * (i + 1)
        ]
        unique_column_ids_i = cur_rank_unique_column_ids.numpy()[
            cur_rank_num_ids * i : cur_rank_num_ids * (i + 1)
        ]
        cur_rank_num_unique_list.append(num_unique_i)
        cur_rank_unique_ids_list.append(np.array(unique_ids_i[0:num_unique_i]))
        cur_rank_unique_column_ids_list.append(
            np.array(unique_column_ids_i[0:num_unique_i])
        )

    global_ids = ids_tensor.numpy()
    np_unique_ids, np_unique_index, np_inverse = np.unique(
        global_ids, return_index=True, return_inverse=True
    )
    np_num_unique = np_unique_ids.size
    # test num unique
    test_case.assertTrue(
        np.array_equal(np_num_unique, np.array(cur_rank_num_unique_list).sum())
    )
    # test unique ids
    unique_ids = np.concatenate(cur_rank_unique_ids_list)
    unique_ids.sort()
    np_unique_ids.sort()
    test_case.assertTrue(np.array_equal(unique_ids, np_unique_ids))
    if has_column_id:
        # test unique column ids
        unique_column_ids = np.concatenate(cur_rank_unique_column_ids_list)
        unique_column_ids.sort()
        global_column_ids = column_ids_tensor.numpy()
        np_unique_column_ids = global_column_ids.flatten()[np_unique_index]
        np_unique_column_ids.sort()
        test_case.assertTrue(np.array_equal(unique_column_ids, np_unique_column_ids))


def _test_embedding_shuffle(test_case, dtype):
    batch_size = int(1024 / parallel_num)
    placement = flow.placement(type="cuda", ranks=list(range(parallel_num)))
    num_columns = 26
    if dtype == flow.float16:
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    data = np.random.rand(max_id, 128).astype(np_dtype)
    data_tensor = flow.tensor(data, requires_grad=False).to_global(
        placement=placement, sbp=flow.sbp.broadcast()
    )

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, column_ids, data):
            (
                num_unique_matrix,
                inverse_unique_partition_indices,
                _,
                cur_rank_unique_ids,
                _,
                cur_rank_inverse_indices,
            ) = flow._C.one_embedding_id_shuffle(ids, column_ids, num_columns)
            unique_embeddings = flow._C.gather(data, cur_rank_unique_ids, axis=0)
            embeddings = flow._C.one_embedding_embedding_shuffle(
                unique_embeddings,
                flow._C.identity(num_unique_matrix),
                flow._C.identity(cur_rank_inverse_indices),
                flow._C.identity(inverse_unique_partition_indices),
            )
            return embeddings

    graph = TestGraph()
    for i in range(10):
        ids_tensor, column_ids_tensor = get_tensors(batch_size, num_columns)
        graph(ids_tensor, column_ids_tensor, data_tensor)
    embeddings = graph(ids_tensor, column_ids_tensor, data_tensor)
    global_ids = ids_tensor.numpy()
    global_data = data_tensor.numpy()
    np_embeddings = global_data[global_ids]
    test_case.assertTrue(np.array_equal(embeddings.numpy(), np_embeddings))


def _test_embedding_gradient_shuffle(test_case):
    batch_size = int(1024 / parallel_num)
    placement = flow.placement(type="cuda", ranks=list(range(parallel_num)))
    num_columns = 26
    embedding_size = 128
    embedding_grad = np.random.rand(batch_size, num_columns, embedding_size).astype(
        np.float32
    )
    embedding_grad_tensor = flow.tensor(embedding_grad, requires_grad=False).to_global(
        placement=placement, sbp=flow.sbp.split(0)
    )

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, column_ids, embedding_grad):
            (
                num_unique_matrix,
                inverse_unique_partition_indices,
                cur_rank_num_unique,
                cur_rank_unique_ids,
                _,
                cur_rank_inverse_indices,
            ) = flow._C.one_embedding_id_shuffle(ids, column_ids, num_columns)
            cur_rank_unique_embedding_grad = flow._C.one_embedding_embedding_gradient_shuffle(
                embedding_grad,
                num_unique_matrix,
                cur_rank_inverse_indices,
                inverse_unique_partition_indices,
            )
            return (
                cur_rank_unique_embedding_grad,
                flow.cast(cur_rank_num_unique, flow.int32),
                cur_rank_unique_ids,
            )

    graph = TestGraph()
    for i in range(10):
        ids_tensor, column_ids_tensor = get_tensors(batch_size, num_columns)
        graph(ids_tensor, column_ids_tensor, embedding_grad_tensor)
    ids_tensor, column_ids_tensor = get_tensors(batch_size, num_columns)
    (
        cur_rank_unique_embedding_grad,
        local_cur_rank_num_unique,
        cur_rank_unique_ids,
    ) = graph(ids_tensor, column_ids_tensor, embedding_grad_tensor)
    cur_rank_num_unique = local_cur_rank_num_unique.to_local().to_global(
        placement=placement, sbp=flow.sbp.split(0)
    )
    global_ids = ids_tensor.numpy()
    global_embedding_grad = embedding_grad_tensor.numpy()
    np_unique_ids = np.unique(global_ids)
    np_num_unique = np_unique_ids.size
    np_cur_rank_unique_embedding_grad = np.zeros((max_id, embedding_size))
    for k in range(np_num_unique):
        unique_id = np_unique_ids[k]
        np_cur_rank_unique_embedding_grad[unique_id, :] = sum(
            global_embedding_grad.reshape(-1, embedding_size)[
                np.where(global_ids.flatten() == unique_id)[0]
            ]
        )

    cur_rank_num_ids = batch_size * num_columns * parallel_num
    of_unique_embedding_grad = np.zeros((max_id, embedding_size))
    for i in range(parallel_num):
        num_unique_i = cur_rank_num_unique.numpy()[i]
        unique_ids_i = cur_rank_unique_ids.numpy()[
            cur_rank_num_ids * i : cur_rank_num_ids * (i + 1)
        ]
        unique_embedding_grad_i = cur_rank_unique_embedding_grad.numpy()[
            cur_rank_num_ids * i : cur_rank_num_ids * (i + 1)
        ]
        for j in range(num_unique_i):
            unique_id = unique_ids_i[j]
            of_unique_embedding_grad[unique_id, :] = unique_embedding_grad_i[j, :]

    test_case.assertTrue(
        np.allclose(
            of_unique_embedding_grad,
            np_cur_rank_unique_embedding_grad,
            atol=1e-4,
            rtol=1e-4,
        ),
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class DataShuffleTestCase(flow.unittest.TestCase):
    def test_id_shuffle(test_case):
        arg_dict = OrderedDict()
        arg_dict["has_column_id"] = [True, False]
        arg_dict["num_columns"] = [1, 26]
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


if __name__ == "__main__":
    unittest.main()
