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

def parseDims(dims, layout):
    if layout == "BHMK":
        B = dims[0]
        H = dims[1]
        M = dims[2]
        K = dims[3]
        merged_dims = dims  # no merge
    elif layout == "BMHK":
        B = dims[0]
        M = dims[1]
        H = dims[2]
        K = dims[3]
        merged_dims = dims
    elif layout == "MBHK":
        B = dims[1]
        M = dims[0]
        H = dims[2]
        K = dims[3]
        merged_dims = dims
    elif layout == "BM(HK)":
        B = dims[0]
        M = dims[1]
        H = dims[2]
        K = dims[3]
        merged_dims = [dims[0], dims[1], dims[2] * dims[3]]
    elif layout == "MB(HK)":
        B = dims[1]
        M = dims[0]
        H = dims[2]
        K = dims[3]
        merged_dims = [dims[0], dims[1], dims[2] * dims[3]]
    
    return B, M, H, K, merged_dims

# all cos&sin are by default in layout (B, H, M, K), in which H is 1
def naive_embedding(x, cos, sin, layout, B, M, H, K, dims, merged_dims):
    y = shuffle_adjacent_two_elem(x, merged_dims)

    if layout == "BHMK":
        naive_out = x * cos + y * sin
    elif layout == "BMHK":
        naive_out = x.reshape(dims) * cos.reshape([B, M, 1, K]) + y.reshape(
            dims
        ) * sin.reshape(
            [B, M, 1, K]
        )  # un-merge
    elif layout == "MBHK":
        naive_out = x.reshape(dims) * cos.transpose([2, 0, 1, 3]).reshape([M, B, 1, K]) + y.reshape(
            dims
        ) * sin.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        )  # un-merge
    elif layout == "BM(HK)":
        naive_out = x.reshape(dims) * cos.reshape([B, M, 1, K]) + y.reshape(
            dims
        ) * sin.reshape(
            [B, M, 1, K]
        )  # un-merge
    elif layout == "MB(HK)":
        naive_out = x.reshape(dims) * cos.transpose([2, 0, 1, 3]).reshape([M, B, 1, K]) + y.reshape(
            dims
        ) * sin.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        )  # un-merge
    
    return naive_out

def _test_without_position(test_case, layout, theta, pass_ndims, dims, dtype):
    print("_test_without_position", layout, theta, pass_ndims, dims, dtype)

    B, M, H, K, merged_dims = parseDims(dims, layout)

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))
    naive_cos = np.array(
        [[
            [math.cos(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    ).reshape(B, 1, M, K)
    naive_sin = np.array(
        [[
            [math.sin(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    ).reshape(B, 1, M, K)

    naive_cos[..., K - pass_ndims:] = 1
    naive_sin[..., K - pass_ndims:] = 0

    naive_out = naive_embedding(x, naive_cos, naive_sin, layout, B, M, H, K, dims, merged_dims)

    fused_cos = np.array(
        [
            [math.cos(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(M)
        ]
    ).reshape(1, 1, M, K)
    fused_sin = np.array(
        [
            [math.sin(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(M)
        ]
    ).reshape(1, 1, M, K)
    fused_x = flow.tensor(x, dtype=dtype, device="cuda")
    fused_cos = flow.tensor(fused_cos.squeeze(), dtype=dtype, device="cuda")
    fused_sin = flow.tensor(fused_sin.squeeze(), dtype=dtype, device="cuda")

    fused_out = flow._C.fused_apply_rotary_emb(fused_x, fused_cos, fused_sin, None, layout, theta, pass_ndims)

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims), fused_out.numpy(), atol=5e-2, rtol=5e-3
        )
    )

def _test_without_position_sinuous(test_case, layout, theta, pass_ndims, dims, dtype):
    print("_test_without_position_sinuous", layout, theta, pass_ndims, dims, dtype)

    B, M, H, K, merged_dims = parseDims(dims, layout)

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))
    naive_cos = np.array(
        [[
            [math.cos(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    ).reshape(B, 1, M, K)
    naive_sin = np.array(
        [[
            [math.sin(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    ).reshape(B, 1, M, K)

    naive_cos[..., K - pass_ndims:] = 1
    naive_sin[..., K - pass_ndims:] = 0

    naive_out = naive_embedding(x, naive_cos, naive_sin, layout, B, M, H, K, dims, merged_dims)

    fused_x = flow.tensor(x, dtype=dtype, device="cuda")

    fused_out = flow._C.fused_apply_rotary_emb(fused_x, None, None, None, layout, theta, pass_ndims)

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims), fused_out.numpy(), atol=5e-2, rtol=5e-3
        )
    )

def _test_with_position_sinuous(test_case, layout, theta, pass_ndims, dims, dtype):
    print("_test_with_position_sinuous", layout, theta, pass_ndims, dims, dtype)

    B, M, H, K, merged_dims = parseDims(dims, layout)

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))

    position_ids = np.random.randint(2*M, size=(B, 2, M), dtype=int)

    naive_cos = np.array(
        [[
            [math.cos(position_ids[b, i // ((K - pass_ndims)//2), m] * (theta ** (2 * (i // 2) / K))) if i < K - pass_ndims else 1 for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    ).reshape(B, 1, M, K)

    naive_sin = np.array(
        [[
            [math.sin(position_ids[b, i // ((K - pass_ndims)//2), m] * (theta ** (2 * (i // 2) / K))) if i < K - pass_ndims else 0 for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    ).reshape(B, 1, M, K)

    naive_cos[..., K - pass_ndims:] = 1
    naive_sin[..., K - pass_ndims:] = 0

    naive_out = naive_embedding(x, naive_cos, naive_sin, layout, B, M, H, K, dims, merged_dims)

    fused_cos = np.array(
        [
            [math.cos(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(2*M)
        ]
    )
    fused_sin = np.array(
        [
            [math.sin(m * (theta ** (2 * (i // 2) / K))) for i in range(K)]
            for m in range(2*M)
        ]
    )

    fused_x = flow.tensor(x, dtype=dtype, device="cuda")
    fused_cos = flow.tensor(fused_cos.squeeze(), dtype=dtype, device="cuda")
    fused_sin = flow.tensor(fused_sin.squeeze(), dtype=dtype, device="cuda")
    fused_position_ids = flow.tensor(position_ids, dtype=flow.int32, device="cuda")

    fused_out = flow._C.fused_apply_rotary_emb(fused_x, fused_cos, fused_sin, fused_position_ids, layout, theta, pass_ndims)

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims), fused_out.numpy(), atol=5e-2, rtol=5e-3
        )
    )

def _test_with_position(test_case, layout, theta, pass_ndims, dims, dtype):
    print("_test_with_position", layout, theta, pass_ndims, dims, dtype)

    B, M, H, K, merged_dims = parseDims(dims, layout)

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))

    position_ids = np.random.randint(2*M, size=(B, 2, M), dtype=int)

    naive_cos = np.array(
        [[
            [math.cos(position_ids[b, i // ((K - pass_ndims)//2), m] * (theta ** (2 * (i // 2) / K))) if i < K - pass_ndims else 1 for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    ).reshape(B, 1, M, K)

    naive_sin = np.array(
        [[
            [math.sin(position_ids[b, i // ((K - pass_ndims)//2), m] * (theta ** (2 * (i // 2) / K))) if i < K - pass_ndims else 0 for i in range(K)]
            for m in range(M)
        ] for b in range(B)]
    ).reshape(B, 1, M, K)

    naive_cos[..., K - pass_ndims:] = 1
    naive_sin[..., K - pass_ndims:] = 0

    naive_out = naive_embedding(x, naive_cos, naive_sin, layout, B, M, H, K, dims, merged_dims)

    fused_x = flow.tensor(x, dtype=dtype, device="cuda")
    fused_position_ids = flow.tensor(position_ids, dtype=flow.int32, device="cuda")

    fused_out = flow._C.fused_apply_rotary_emb(fused_x, None, None, fused_position_ids, layout, theta, pass_ndims)

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims), fused_out.numpy(), atol=5e-2, rtol=5e-3
        )
    )

'''
1. if cos&sin is given, then theta will not be used
2. if cos&sin is not given, then any form of layout which cannot infer the dimension of k is not allowed, e.g. BM(HK)
2. if position_ids is given, then M of cos&sin could be different from M of x
'''

@flow.unittest.skip_unless_1n1d()
class TestFusedRotaryEmbedding(flow.unittest.TestCase):
    # because rule no.2, kernels without cos&sin cannot work under specific layout
    def test_fused_rotary_embedding_op_with_sinuous(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_with_position_sinuous, _test_without_position]
        args_dict["layout"] = ["BMHK", "BHMK", "MBHK", "BM(HK)","MB(HK)"]
        args_dict["theta"] = [1e-1]
        args_dict["pass_ndims"] = [0, 4]
        args_dict["dims"] = [(2,8,3,8)]
        #args_dict["pass_ndims"] = [48]
        #args_dict["dims"] = [(32, 2048, 32, 64)]
        args_dict["dtype"] = [flow.float16]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])

    def test_fused_rotary_embedding_op_without_sinuous(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_with_position, _test_without_position_sinuous]
        args_dict["layout"] = ["BMHK", "BHMK", "MBHK"]
        args_dict["theta"] = [1e-1]
        args_dict["pass_ndims"] = [0, 4]
        args_dict["dims"] = [(2,8,3,8)]
        #args_dict["pass_ndims"] = [48]
        #args_dict["dims"] = [(32, 2048, 32, 64)]
        args_dict["dtype"] = [flow.float16, flow.float32]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
