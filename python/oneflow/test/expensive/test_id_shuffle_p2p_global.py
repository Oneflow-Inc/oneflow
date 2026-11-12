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

os.environ["ONEFLOW_ONE_EMBEDDING_USE_DYNAMIC_MEMORY_ALLOCATION"] = "0"
os.environ["ONEFLOW_ONE_EMBEDDING_ID_SHUFFLE_USE_P2P"] = "1"
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest

parallel_num = 2
max_id = 1000
num_tables = 26
batch_size = 64


def _test_id_shuffle_p2p(test_case):
    placement = flow.placement(type="cuda", ranks=list(range(parallel_num)))

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
            ) = flow._C.one_embedding_id_shuffle(ids, table_ids, num_tables, "test")
            return (
                flow.cast(num_unique_matrix, flow.int32),
                flow.cast(cur_rank_num_unique, flow.int32),
                cur_rank_unique_ids,
                flow.cast(cur_rank_unique_table_ids, flow.int32),
            )

    rank = flow.env.get_rank()
    graph = TestGraph()
    # The P2P kernel shares its buffers with the peer once and reuses them, so check
    # every iteration, not only the last one.
    for i in range(10):
        ids = np.random.randint(0, max_id, (batch_size, num_tables), dtype=np.int64)
        ids_tensor = flow.tensor(ids).to_global(
            placement=placement, sbp=flow.sbp.split(0)
        )
        table_ids_tensor = flow.tensor((ids % num_tables).astype(np.int32)).to_global(
            placement=placement, sbp=flow.sbp.split(0)
        )
        (
            num_unique_matrix,
            cur_rank_num_unique,
            cur_rank_unique_ids,
            cur_rank_unique_table_ids,
        ) = graph(ids_tensor, table_ids_tensor)
        global_ids = ids_tensor.numpy().reshape(parallel_num, -1)
        num_unique = int(cur_rank_num_unique.to_local().numpy()[0])
        unique_ids = cur_rank_unique_ids.to_local().numpy()[:num_unique]
        unique_table_ids = cur_rank_unique_table_ids.to_local().numpy()[:num_unique]
        # This rank owns the ids that hash to it: a set of distinct ids from the
        # batch, with their table ids.
        test_case.assertEqual(np.unique(unique_ids).size, num_unique)
        test_case.assertTrue(np.isin(unique_ids, global_ids).all())
        test_case.assertTrue(np.array_equal(unique_table_ids, unique_ids % num_tables))
        # num_unique_matrix[r][p] counts rank r's distinct ids owned by rank p.
        matrix = num_unique_matrix.to_local().numpy().reshape(parallel_num, -1)
        expected_column = [
            np.isin(np.unique(global_ids[r]), unique_ids).sum()
            for r in range(parallel_num)
        ]
        test_case.assertTrue(
            np.array_equal(matrix[:, rank], expected_column),
            "num_unique_matrix mismatch at iteration %d" % i,
        )
        # Together the ranks own every distinct id exactly once.
        all_num_unique = (
            cur_rank_num_unique.to_local()
            .to_global(placement=placement, sbp=flow.sbp.split(0))
            .numpy()
        )
        test_case.assertEqual(all_num_unique.sum(), np.unique(global_ids).size)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class IdShuffleP2PTestCase(flow.unittest.TestCase):
    def test_id_shuffle_p2p(test_case):
        # One graph per process: the P2P kernel exchanges buffers under its op
        # name, which a second graph would reuse.
        _test_id_shuffle_p2p(test_case)


if __name__ == "__main__":
    unittest.main()
