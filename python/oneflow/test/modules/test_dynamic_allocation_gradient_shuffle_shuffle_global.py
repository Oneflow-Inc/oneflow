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
os.environ["ONEFLOW_ONE_EMBEDDING_USE_DYNAMIC_MEMORY_ALLOCATION"] = "1"
import unittest
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict
import numpy as np
import oneflow as flow

parallel_num = 2
max_id = 1000


def get_tensors(batch_size, num_tables):
    placement = flow.placement(type="cuda", ranks=list(range(parallel_num)))
    ids = np.random.randint(0, max_id, (batch_size, num_tables), dtype=np.int64)
    ids_tensor = flow.tensor(ids, requires_grad=False).to_global(
        placement=placement, sbp=flow.sbp.split(0)
    )
    table_ids = (
        ids % num_tables
    )  # same id must have same table id, so in this case get table_ids from ids
    table_ids_tensor = flow.tensor(
        table_ids.astype(np.int32), requires_grad=False
    ).to_global(placement=placement, sbp=flow.sbp.split(0))
    return ids_tensor, table_ids_tensor


def round_half_away_from_zero(x):
    sign = np.sign(x)
    abs_val = np.abs(x)
    abs_val += 0.5
    floor_val = np.floor(abs_val)
    out = floor_val * sign
    return out


def _test_embedding_gradient_shuffle(test_case, enable_quantize, fp16, embedding_size):
    np_tolerance = 0
    batch_size = int(1024 / parallel_num)
    placement = flow.placement(type="cuda", ranks=list(range(parallel_num)))
    num_tables = 26
    enable_quantized_comm = enable_quantize and embedding_size < 1025
    if enable_quantized_comm:
        np_tolerance = 0.5
        os.environ["ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM"] = "1"
    else:
        if fp16:
            np_tolerance = 1e-2
        else:
            np_tolerance = 1e-4
        os.environ["ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM"] = "0"
    embedding_grad = np.random.rand(batch_size, num_tables, embedding_size).astype(
        np.float32
    )
    embedding_grad_tensor = flow.tensor(embedding_grad, requires_grad=False).to_global(
        placement=placement, sbp=flow.sbp.split(0)
    )

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, ids, table_ids, embedding_grad):
            (
                num_unique_matrix,
                inverse_unique_partition_indices,
                cur_rank_num_unique,
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
                flow.cast(cur_rank_num_unique, flow.int32),
                cur_rank_unique_ids,
            )

    graph = TestGraph()
    for i in range(10):
        ids_tensor, table_ids_tensor = get_tensors(batch_size, num_tables)
        graph(ids_tensor, table_ids_tensor, embedding_grad_tensor)
    ids_tensor, table_ids_tensor = get_tensors(batch_size, num_tables)
    (
        cur_rank_unique_embedding_grad,
        local_cur_rank_num_unique,
        cur_rank_unique_ids,
    ) = graph(ids_tensor, table_ids_tensor, embedding_grad_tensor)
    cur_rank_num_unique = local_cur_rank_num_unique.to_local().to_global(
        placement=placement, sbp=flow.sbp.split(0)
    )
    global_ids = ids_tensor.numpy()
    global_embedding_grad = embedding_grad_tensor.numpy()
    np_unique_ids = np.unique(global_ids)
    np_num_unique = np_unique_ids.size
    np_cur_rank_unique_embedding_grad = np.zeros((max_id, embedding_size))
    if fp16:
        global_embedding_grad = global_embedding_grad.astype(np.float16)
    for k in range(np_num_unique):
        unique_id = np_unique_ids[k]
        np_data = sum(
            global_embedding_grad.reshape(-1, embedding_size)[
                np.where(global_ids.flatten() == unique_id)[0]
            ]
        )
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

        np_cur_rank_unique_embedding_grad[unique_id, :] = np_data
        if fp16:
            np_cur_rank_unique_embedding_grad = np_cur_rank_unique_embedding_grad.astype(
                np.float32
            )

    cur_rank_num_ids = batch_size * num_tables * parallel_num
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
            atol=np_tolerance,
            rtol=np_tolerance,
        ),
    )


# FIXME: restore this test after upgrading CUDA driver
@unittest.skip("CUDA driver version of CI machine is insufficient for this test")
# @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class DataShuffleTestCase(flow.unittest.TestCase):
    def test_embedding_gradient_shuffle(test_case):
        arg_dict = OrderedDict()
        arg_dict["enable_quantize"] = [True, False]
        arg_dict["fp16"] = [True, False]
        arg_dict["embedding_size"] = [128, 17]
        for kwargs in GenArgDict(arg_dict):
            _test_embedding_gradient_shuffle(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
