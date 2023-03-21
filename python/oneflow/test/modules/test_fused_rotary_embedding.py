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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import numpy as np
import math


def shuffle_adjacent_two_elem(x, dims):
    y = x.copy()
    num_elements = 1
    stride = [1] * len(dims)

    for i in range(len(dims)):
        dim = dims[len(dims) - 1 - i]
        num_elements = num_elements * dim
        if i > 0:
            stride[len(dims) - 1 - i] = stride[len(dims) - i] * dims[len(dims) - i]

    for i in range(len(dims) - 1):
        stride[i] = stride[i] // 2
    num_elements = num_elements // 2

    for i in range(num_elements):
        index = [1] * len(dims)
        offset = i
        for j in range(len(stride)):
            s = stride[j]
            index[j] = offset // s
            offset = offset - index[j] * s

        index[-1] = index[-1] * 2
        switch_index = index.copy()
        switch_index[-1] = switch_index[-1] + 1
        y[(*index,)], y[(*switch_index,)] = -y[(*switch_index,)], y[(*index,)]

    return y


def _test_fused_rotary_embedding(test_case, layout, pass_ndims, dims, dtype):
    theta = 1e-1

    if layout == "BHMK":
        B = dims[0]
        H = dims[1]
        M = dims[2]
        K = dims[3]
        merged_dims = dims  # no merge
    else:
        B = dims[0]
        M = dims[1]
        H = dims[2]
        K = dims[3]
        merged_dims = [dims[0], dims[1], dims[2] * dims[3]]

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))
    cos = np.array(
        [
            [math.cos(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(2*M)
        ]
    )
    sin = np.array(
        [
            [math.sin(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(2*M)
        ]
    )

    position_ids = np.random.randint(M, size=(B, 2, M), dtype=int)

    #position_ids = np.array([[[m for i in range(2)] for m in range(M)] for b in range(B)])

    y = shuffle_adjacent_two_elem(x, merged_dims)

    naive_cos = np.array(
        [[
            [math.cos(position_ids[b, i // ((K - pass_ndims)//2), m] * (theta ** (2 * (i // 2) / K))) if i < K - pass_ndims else 1 for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    )

    naive_sin = np.array(
        [[
            [math.sin(position_ids[b, i // ((K - pass_ndims)//2), m] * (theta ** (2 * (i // 2) / K))) if i < K - pass_ndims else 0 for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    )


    if layout == "BHMK":
        naive_out = x * naive_cos.reshape([B, 1, M, K]) + y * naive_sin.reshape([B, 1, M, K])
    else:
        naive_out = x.reshape(dims) * naive_cos.reshape([B, M, 1, K]) + y.reshape(
            dims
        ) * naive_sin.reshape(
            [B, M, 1, K]
        )  # un-merge

    fused_x = flow.tensor(x, dtype=dtype, device="cuda")
    fused_cos = flow.tensor(cos, dtype=dtype, device="cuda")
    fused_sin = flow.tensor(sin, dtype=dtype, device="cuda")
    fused_position_ids = flow.tensor(position_ids, dtype=flow.int32, device="cuda")

    fused_out = flow._C.fused_apply_rotary_emb(fused_x, None, None, fused_position_ids, layout, theta, pass_ndims)

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims), fused_out.numpy(), atol=5e-2, rtol=5e-3
        )
    )

@flow.unittest.skip_unless_1n1d()
class TestFusedRotaryEmbedding(flow.unittest.TestCase):
    def test_fused_rotary_embedding_op(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_fused_rotary_embedding]
        args_dict["layout"] = ["BHMK"]
        args_dict["pass_ndims"] = [48]
        args_dict["dims"] = [(1,1,1,64), (1,1,5,64), (1,3,1,64), (1,3,5,64), (2,1,1,64), (2,1,5,64), (2,3,1,64), (2,3,5,64)]
        #args_dict["dims"] = [(32, 2048, 32, 64)]
        args_dict["dtype"] = [flow.float16]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
