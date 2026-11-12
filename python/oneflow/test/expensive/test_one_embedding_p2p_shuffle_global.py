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

# The P2P kernels are chosen from these switches, which are read once per process.
os.environ["ONEFLOW_ONE_EMBEDDING_USE_DYNAMIC_MEMORY_ALLOCATION"] = "0"
os.environ["ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION"] = "1"
os.environ["ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_USE_P2P"] = "1"
os.environ["ONEFLOW_ONE_EMBEDDING_EMBEDDING_GRADIENT_SHUFFLE_USE_P2P"] = "1"
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest

parallel_num = 2
max_id = 1000
num_tables = 26
embedding_size = 128
# Small enough that the shared buffers stay in a GPU cache between iterations.
batch_size = 64


def _build_op(op_type_name, inputs, output, attrs):
    builder = flow._oneflow_internal.one.OpBuilder(
        op_type_name, flow.framework.id_util.UniqueStr(op_type_name)
    )
    for name in inputs:
        builder.input(name, 1)
    builder.output(output, 1)
    for key, value in attrs.items():
        builder.attr(key, value)
    return builder.build()


def _apply(op, inputs):
    # Any dispatcher taking a TensorTuple applies the op; its extra attrs are not
    # attrs of these ops and are ignored.
    return flow._C.dispatch_crop_mirror_normalize_from_uint8(
        op, inputs, mean=[0.0], std=[1.0]
    )


def _test_p2p_embedding_shuffle(test_case):
    # skip_last_gather and skip_first_scatter select the P2P kernels; only the fused
    # embedding interaction pass sets them, so the ops are built directly here.
    placement = flow.placement(type="cuda", ranks=list(range(parallel_num)))
    shuffle_op = _build_op(
        "embedding_shuffle",
        [
            "cur_rank_embeddings",
            "num_unique_matrix",
            "cur_rank_inverse_indices",
            "inverse_unique_partition_indices",
        ],
        "embeddings",
        {
            "embedding_size": "at_int64: %d" % embedding_size,
            "skip_last_gather": "at_bool: true",
            "is_train": "at_bool: true",
            "embedding_name": 'at_string: "test"',
        },
    )
    gradient_shuffle_op = _build_op(
        "embedding_gradient_shuffle",
        [
            "embedding_grad",
            "num_unique_matrix",
            "cur_rank_inverse_indices",
            "inverse_unique_partition_indices",
        ],
        "cur_rank_unique_embedding_grad",
        {
            "embedding_size": "at_int64: %d" % embedding_size,
            "skip_first_scatter": "at_bool: true",
            "embedding_name": 'at_string: "test"',
        },
    )

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, data):
            (
                num_unique_matrix,
                inverse_unique_partition_indices,
                cur_rank_num_unique,
                cur_rank_unique_ids,
                _,
                cur_rank_inverse_indices,
            ) = flow._C.one_embedding_id_shuffle(ids, None, num_tables, "test")
            # The P2P kernels share these buffers with the peer once, so they must
            # keep their addresses across iterations.
            num_unique_matrix = flow._C.identity(num_unique_matrix)
            cur_rank_inverse_indices = flow._C.identity(cur_rank_inverse_indices)
            inverse_unique_partition_indices = flow._C.identity(
                inverse_unique_partition_indices
            )
            unique_embeddings = flow._C.identity(
                flow._C.gather(data, cur_rank_unique_ids, axis=0)
            )
            embeddings = _apply(
                shuffle_op,
                [
                    unique_embeddings,
                    num_unique_matrix,
                    cur_rank_inverse_indices,
                    inverse_unique_partition_indices,
                ],
            )
            unique_embedding_grad = _apply(
                gradient_shuffle_op,
                [
                    embeddings,
                    num_unique_matrix,
                    cur_rank_inverse_indices,
                    inverse_unique_partition_indices,
                ],
            )
            return (
                embeddings,
                unique_embedding_grad,
                flow.cast(inverse_unique_partition_indices, flow.int32),
                flow.cast(cur_rank_num_unique, flow.int32),
                cur_rank_unique_ids,
            )

    rank = flow.env.get_rank()
    graph = TestGraph()
    for i in range(20):
        # The table changes every iteration, so a peer reading a previous
        # iteration's rows gives a wrong answer. Multiples of 1/8 keep float16
        # sums exact.
        data = (
            np.random.RandomState(i).randint(-64, 64, (max_id, embedding_size)) / 8
        ).astype(np.float16)
        data_tensor = flow.tensor(data).to_global(
            placement=placement, sbp=flow.sbp.broadcast()
        )
        ids = np.random.randint(0, max_id, (batch_size, num_tables), dtype=np.int64)
        ids_tensor = flow.tensor(ids).to_global(
            placement=placement, sbp=flow.sbp.split(0)
        )
        (
            embeddings,
            unique_embedding_grad,
            inverse_unique_partition_indices,
            cur_rank_num_unique,
            cur_rank_unique_ids,
        ) = graph(ids_tensor, data_tensor)
        global_ids = ids_tensor.numpy().reshape(parallel_num, -1)
        local_ids = global_ids[rank]
        # Forward: each id's row, found through inverse_unique_partition_indices.
        local_embeddings = embeddings.to_local().numpy().reshape(-1, embedding_size)
        inverse = inverse_unique_partition_indices.to_local().numpy().flatten()
        test_case.assertTrue(
            np.array_equal(local_embeddings[inverse], data[local_ids]),
            "forward mismatch at iteration %d" % i,
        )
        # Backward, fed the forward output: each unique id's row times the number
        # of ranks that hold the id.
        num_unique = int(cur_rank_num_unique.to_local().numpy()[0])
        unique_ids = cur_rank_unique_ids.to_local().numpy()[:num_unique]
        holders = sum(np.isin(unique_ids, global_ids[r]) for r in range(parallel_num))
        expected = data[unique_ids].astype(np.float32) * holders[:, None]
        grad = (
            unique_embedding_grad.to_local()
            .numpy()
            .reshape(-1, embedding_size)[:num_unique]
        )
        test_case.assertTrue(
            np.array_equal(grad.astype(np.float32), expected),
            "backward mismatch at iteration %d" % i,
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class P2PShuffleTestCase(flow.unittest.TestCase):
    def test_p2p_embedding_shuffle(test_case):
        # One graph per process: the P2P kernels exchange buffers under their op
        # names, which a second graph would reuse.
        _test_p2p_embedding_shuffle(test_case)


if __name__ == "__main__":
    unittest.main()
