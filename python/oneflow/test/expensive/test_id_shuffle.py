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

# dynamic memory allocation can't be tested in unittest
os.environ["ONEFLOW_ONE_EMBEDDING_USE_DYNAMIC_MEMORY_ALLOCATION"] = "0"
import unittest
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict
import numpy as np
import oneflow as flow


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
            ) = flow._C.one_embedding_id_shuffle(ids, table_ids, num_tables, "test")
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


def round_half_away_from_zero(x):
    sign = np.sign(x)
    abs_val = np.abs(x)
    abs_val += 0.5
    floor_val = np.floor(abs_val)
    out = floor_val * sign
    return out


def embedding_shuffle_quantize(np_data, np_dtype):
    # When use float16, ComputeType is set to as Float.
    np_reduce_data = np_data.astype(np.float32)
    abs_max_factor = np.max(np.abs(np_reduce_data), axis=2)
    abs_max_factor = np.expand_dims(abs_max_factor, axis=2)
    transport_quantize_factor = abs_max_factor.astype(np_dtype)
    int8_factor = np.ones(abs_max_factor.shape, dtype=np.float32) * 127.0
    int8_factor = int8_factor.astype(np.float32)
    quantize_factor = int8_factor / abs_max_factor

    # Covert to Compute Type.
    np_data.astype(np.float32)
    np_data = np_data * quantize_factor
    np_data = round_half_away_from_zero(np_data)
    np_data = np_data.astype(np.int8)

    # Covert to Compute Type.
    np_data = np_data.astype(np.float32)
    dequantize_factor = transport_quantize_factor.astype(np.float32) / int8_factor
    np_data = np_data * dequantize_factor
    np_data = np_data.astype(np_dtype)
    return np_data


def _test_embedding_shuffle(test_case, dtype, enable_quantize):
    batch_size = 512
    num_tables = 26
    embedding_size = 128
    ids = np.random.randint(0, 1000, (batch_size, num_tables), dtype=np.int64)

    enable_quantized_comm = enable_quantize and embedding_size < 1025
    if enable_quantized_comm:
        os.environ["ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM"] = "1"
    else:
        os.environ["ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM"] = "0"

    table_ids = (
        ids % num_tables
    )  # same id must have same table id, so in this case get table_ids from ids
    if dtype == flow.float16:
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    data = np.random.rand(1000, embedding_size).astype(np_dtype)

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
            ) = flow._C.one_embedding_id_shuffle(ids, table_ids, num_tables, "test")
            unique_embeddings = flow._C.gather(data, cur_rank_unique_ids, axis=0)
            embeddings = flow._C.one_embedding_embedding_shuffle(
                unique_embeddings,
                num_unique_matrix,
                cur_rank_inverse_indices,
                inverse_unique_partition_indices,
                "test",
            )
            return embeddings

    graph = TestGraph()
    embeddings = graph(ids_tensor, table_ids_tensor, data_tensor)
    np_embeddings = data[ids]

    # Quantized numpy embedding.
    if enable_quantized_comm:
        np_embeddings = embedding_shuffle_quantize(np_embeddings, np_dtype)
    test_case.assertTrue(
        np.allclose(embeddings.numpy(), np_embeddings, atol=1e-4, rtol=1e-4)
    )


def _test_embedding_gradient_shuffle(test_case, enable_quantize, fp16, embedding_size):
    batch_size = 512
    num_tables = 26
    ids = np.random.randint(0, 1000, (batch_size, num_tables), dtype=np.int64)
    enable_quantized_comm = enable_quantize and embedding_size < 1025
    if enable_quantized_comm:
        np_tolerance = 0.5
        os.environ["ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM"] = "1"
        ids = np.arange(batch_size * num_tables, dtype=np.int64)
        np.random.shuffle(ids)
    else:
        if fp16:
            np_tolerance = 1e-2
        else:
            np_tolerance = 1e-4
        os.environ["ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM"] = "0"

    table_ids = (
        ids % num_tables
    )  # same id must have same table id, so in this case get table_ids from ids
    embedding_grad = np.random.uniform(
        low=-1, high=1, size=(batch_size, num_tables, embedding_size)
    ).astype(np.float32)
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
            ) = flow._C.one_embedding_id_shuffle(ids, table_ids, num_tables, "test")
            if fp16:
                embedding_grad = flow.cast(embedding_grad, flow.float16)
            cur_rank_unique_embedding_grad = flow._C.one_embedding_embedding_gradient_shuffle(
                embedding_grad,
                num_unique_matrix,
                cur_rank_inverse_indices,
                inverse_unique_partition_indices,
                "test",
            )
            if fp16:
                cur_rank_unique_embedding_grad = flow.cast(
                    cur_rank_unique_embedding_grad, flow.float32
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
        cur_rank_unique_embedding_grad.shape, dtype=np.float32
    ).reshape(-1, embedding_size)

    embedding_grad = embedding_grad.reshape(-1, embedding_size)
    if fp16:
        embedding_grad = embedding_grad.astype(np.float16)
    for k in range(np_num_unique):
        np_data = sum(embedding_grad[np.where(ids.flatten() == np_unique_ids[k])[0]])
        # Quantize Embedding Gradient.
        if enable_quantized_comm:
            abs_max_factor = np.max(np.abs(np_data))
            int8_factor = np.full(abs_max_factor.shape, 127.0, dtype=np.float32)
            quantize_factor = int8_factor / abs_max_factor
            np_data = np_data * quantize_factor
            np_data = round_half_away_from_zero(np_data)
            np_data = np_data.astype(np.int8)
            np_data = np_data.astype(np.float32)
            dequantize_factor = abs_max_factor / int8_factor
            np_data = np_data * dequantize_factor

        np_cur_rank_unique_embedding_grad[k, :] = np_data

    reversed_ids = cur_rank_unique_ids[cur_rank_inverse_indices][
        inverse_unique_partition_indices
    ]
    test_case.assertTrue(np.array_equal(reversed_ids.numpy(), ids))
    of_cur_rank_embedding_grad = cur_rank_unique_embedding_grad[
        cur_rank_inverse_indices
    ][inverse_unique_partition_indices]
    of_cur_rank_embedding_grad = flow.reshape(
        of_cur_rank_embedding_grad, (-1, embedding_size)
    )
    np_cur_rank_embedding_grad = np_cur_rank_unique_embedding_grad[np_inverse]
    if fp16:
        np_cur_rank_embedding_grad = np_cur_rank_embedding_grad.astype(np.float32)

    test_case.assertTrue(
        np.allclose(
            of_cur_rank_embedding_grad.numpy().flatten(),
            np_cur_rank_embedding_grad.flatten(),
            atol=np_tolerance,
            rtol=np_tolerance,
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
            ) = flow._C.one_embedding_unique_key_value_pair(
                ids, table_ids, num_tables, "test"
            )
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
        arg_dict["enable_quantize"] = [True, False]

        for kwargs in GenArgDict(arg_dict):
            _test_embedding_shuffle(test_case, **kwargs)

    def test_embedding_gradient_shuffle(test_case):
        arg_dict = OrderedDict()
        arg_dict["enable_quantize"] = [True, False]
        arg_dict["fp16"] = [True, False]
        arg_dict["embedding_size"] = [128, 17]
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
