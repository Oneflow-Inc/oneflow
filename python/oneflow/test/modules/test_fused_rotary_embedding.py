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


def plane_shuffle(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return np.concatenate((-x2, x1), axis=-1)


def shuffle_adjacent_two_elem(x):
    y = x.copy()
    for i in range(x.shape[-1] // 2):
        y[..., 2 * i] = -x[..., 2 * i + 1]
        y[..., 2 * i + 1] = x[..., 2 * i]
    return y


def parseDims(dims, x_layout):
    B = 1
    M = 1
    H = 1
    K = 1
    merged_dims = dims
    if x_layout == "BHMK":
        B = dims[0]
        H = dims[1]
        M = dims[2]
        K = dims[3]
        merged_dims = dims  # no merge
    elif x_layout == "BMHK":
        B = dims[0]
        M = dims[1]
        H = dims[2]
        K = dims[3]
        merged_dims = dims
    elif x_layout == "MBHK":
        B = dims[1]
        M = dims[0]
        H = dims[2]
        K = dims[3]
        merged_dims = dims
    elif x_layout == "BM(HK)":
        B = dims[0]
        M = dims[1]
        H = dims[2]
        K = dims[3]
        merged_dims = [dims[0], dims[1], dims[2] * dims[3]]
    elif x_layout == "MB(HK)":
        B = dims[1]
        M = dims[0]
        H = dims[2]
        K = dims[3]
        merged_dims = [dims[0], dims[1], dims[2] * dims[3]]
    elif x_layout == "BM(H3K)":
        B = dims[0]
        M = dims[1]
        H = dims[2]
        K = dims[3]
        merged_dims = [dims[0], dims[1], 3 * dims[2] * dims[3]]
    elif x_layout == "MB(H3K)":
        B = dims[1]
        M = dims[0]
        H = dims[2]
        K = dims[3]
        merged_dims = [dims[0], dims[1], 3 * dims[2] * dims[3]]

    return B, M, H, K, merged_dims


# all cos&sin are by default in x_layout (B, H, M, K), in which H is 1
def naive_embedding(
    x,
    cos,
    sin,
    x_layout,
    B,
    M,
    H,
    K,
    dims,
    merged_dims,
    rotary_size,
    rotary_ndims,
    mode,
):
    naive_out = None
    if mode == "plane":
        if rotary_ndims == 2:
            y1 = plane_shuffle(x[..., : rotary_size // 2])
            y2 = plane_shuffle(x[..., rotary_size // 2 : rotary_size])
            y3 = x[..., rotary_size:]
            y = np.concatenate((y1, y2, y3), axis=-1)
        else:
            y1 = plane_shuffle(x[..., :rotary_size])
            y2 = x[..., rotary_size:]
            y = np.concatenate((y1, y2), axis=-1)
    else:
        y = shuffle_adjacent_two_elem(x)

    if x_layout == "BHMK":
        naive_out = x * cos + y * sin
    elif x_layout == "BMHK":
        naive_out = x.reshape(dims) * cos.reshape([B, M, 1, K]) + y.reshape(
            dims
        ) * sin.reshape(
            [B, M, 1, K]
        )  # un-merge
    elif x_layout == "MBHK" or x_layout == "MB(HK)":
        naive_out = x.reshape(dims) * cos.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        ) + y.reshape(dims) * sin.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        )  # un-merge
    elif x_layout == "BM(HK)":
        naive_out = x.reshape(dims) * cos.reshape([B, M, 1, K]) + y.reshape(
            dims
        ) * sin.reshape(
            [B, M, 1, K]
        )  # un-merge
    elif x_layout == "BM(H3K)":
        out0 = x[..., 0, :].reshape(dims) * cos.reshape([B, M, 1, K]) + y[
            ..., 0, :
        ].reshape(dims) * sin.reshape([B, M, 1, K])
        out1 = x[..., 1, :].reshape(dims) * cos.reshape([B, M, 1, K]) + y[
            ..., 1, :
        ].reshape(dims) * sin.reshape([B, M, 1, K])
        out2 = x[..., 2, :].reshape(dims) * cos.reshape([B, M, 1, K]) + y[
            ..., 2, :
        ].reshape(dims) * sin.reshape([B, M, 1, K])

        naive_out = np.concatenate((out0, out1, out2), axis=-1)
    elif x_layout == "MB(H3K)":
        out0 = x[..., 0, :].reshape(dims) * cos.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        ) + y[..., 0, :].reshape(dims) * sin.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        )
        out1 = x[..., 1, :].reshape(dims) * cos.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        ) + y[..., 1, :].reshape(dims) * sin.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        )
        out2 = x[..., 2, :].reshape(dims) * cos.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        ) + y[..., 2, :].reshape(dims) * sin.transpose([2, 0, 1, 3]).reshape(
            [M, B, 1, K]
        )

        naive_out = np.concatenate((out0, out1, out2), axis=-1)

    return naive_out


# this assume that rotary_ndims is by default 1
def _test_without_position(
    test_case, x_layout, mode, base, rotary_size, dims, rotary_ndims, dtype, device
):
    B, M, H, K, merged_dims = parseDims(dims, x_layout)

    np.random.seed(3124)

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))
    naive_cos = np.array(
        [
            [
                [
                    math.cos(
                        m
                        * (
                            (1 / base)
                            ** (
                                2
                                * ((i % (rotary_size / rotary_ndims)) // 2)
                                / (rotary_size / rotary_ndims)
                            )
                        )
                    )
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)
    naive_sin = np.array(
        [
            [
                [
                    math.sin(
                        m
                        * (
                            (1 / base)
                            ** (
                                2
                                * ((i % (rotary_size / rotary_ndims)) // 2)
                                / (rotary_size / rotary_ndims)
                            )
                        )
                    )
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)

    naive_cos[..., rotary_size:] = 1
    naive_sin[..., rotary_size:] = 0

    naive_x = x
    if x_layout == "BM(HK)" or x_layout == "BM(H2K)" or x_layout == "BM(H3K)":
        naive_x = x.reshape([B, M, H, -1, K])
    elif x_layout == "MB(HK)" or x_layout == "MB(H2K)" or x_layout == "MB(H3K)":
        naive_x = x.reshape([M, B, H, -1, K])

    naive_out = naive_embedding(
        naive_x,
        naive_cos,
        naive_sin,
        x_layout,
        B,
        M,
        H,
        K,
        dims,
        merged_dims,
        rotary_size,
        rotary_ndims,
        mode,
    )

    fused_cos = np.array(
        [
            [
                math.cos(
                    m
                    * (
                        (1 / base)
                        ** (
                            2
                            * ((i % (rotary_size // rotary_ndims)) // 2)
                            / (rotary_size / rotary_ndims)
                        )
                    )
                )
                for i in range(rotary_size // rotary_ndims)
            ]
            for m in range(M)
        ]
    ).reshape(M, rotary_size // rotary_ndims)
    fused_sin = np.array(
        [
            [
                math.sin(
                    m
                    * (
                        (1 / base)
                        ** (
                            2
                            * ((i % (rotary_size // rotary_ndims)) // 2)
                            / (rotary_size // rotary_ndims)
                        )
                    )
                )
                for i in range(rotary_size // rotary_ndims)
            ]
            for m in range(M)
        ]
    ).reshape(M, rotary_size // rotary_ndims)
    fused_x = flow.tensor(x, dtype=dtype, device=device)
    fused_cos = flow.tensor(fused_cos, dtype=dtype, device=device)
    fused_sin = flow.tensor(fused_sin, dtype=dtype, device=device)

    if x_layout == "BM(H3K)":
        out0 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=fused_cos,
            sin=fused_sin,
            position_ids=None,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=0,
        )
        out1 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=fused_cos,
            sin=fused_sin,
            position_ids=None,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=1,
        )
        out2 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=fused_cos,
            sin=fused_sin,
            position_ids=None,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=2,
        )

        fused_out = np.concatenate((out0, out1, out2), axis=-1)
    else:
        fused_out = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=fused_cos,
            sin=fused_sin,
            position_ids=None,
            x_layout=x_layout,
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
        ).numpy()

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims),
            fused_out.reshape(merged_dims),
            atol=5e-2,
            rtol=5e-3,
        )
    )


# this assume that rotary_ndims is by default 1
def _test_without_position_sinuous(
    test_case, x_layout, mode, base, rotary_size, dims, rotary_ndims, dtype, device
):
    B, M, H, K, merged_dims = parseDims(dims, x_layout)

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))
    naive_cos = np.array(
        [
            [
                [
                    math.cos(
                        m
                        * (
                            (1 / base)
                            ** (
                                2
                                * ((i % (rotary_size // rotary_ndims)) // 2)
                                / (rotary_size // rotary_ndims)
                            )
                        )
                    )
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)
    naive_sin = np.array(
        [
            [
                [
                    math.sin(
                        m
                        * (
                            (1 / base)
                            ** (
                                2
                                * ((i % (rotary_size // rotary_ndims)) // 2)
                                / (rotary_size // rotary_ndims)
                            )
                        )
                    )
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)

    naive_cos[..., rotary_size:] = 1
    naive_sin[..., rotary_size:] = 0

    naive_x = x
    if x_layout == "BM(HK)" or x_layout == "BM(H2K)" or x_layout == "BM(H3K)":
        naive_x = x.reshape([B, M, H, -1, K])
    elif x_layout == "MB(HK)" or x_layout == "MB(H2K)" or x_layout == "MB(H3K)":
        naive_x = x.reshape([M, B, H, -1, K])

    naive_out = naive_embedding(
        naive_x,
        naive_cos,
        naive_sin,
        x_layout,
        B,
        M,
        H,
        K,
        dims,
        merged_dims,
        rotary_size,
        rotary_ndims,
        mode,
    )

    fused_x = flow.tensor(x, dtype=dtype, device=device)

    if x_layout == "BM(H3K)":
        out0 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=None,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=0,
        )
        out1 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=None,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=1,
        )
        out2 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=None,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=2,
        )

        fused_out = np.concatenate((out0, out1, out2), axis=-1)
    else:
        fused_out = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=None,
            x_layout=x_layout,
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
        ).numpy()

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims),
            fused_out.reshape(merged_dims),
            atol=5e-2,
            rtol=5e-3,
        )
    )


def _test_with_position_sinuous(
    test_case, x_layout, mode, base, rotary_size, dims, rotary_ndims, dtype, device
):
    B, M, H, K, merged_dims = parseDims(dims, x_layout)

    np.random.seed(3124)

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))

    position_ids = np.random.randint(2 * M, size=(B, rotary_ndims, M), dtype=np.int64)

    naive_cos = np.array(
        [
            [
                [
                    math.cos(
                        position_ids[b, i // ((rotary_size) // rotary_ndims), m]
                        * (
                            (1 / base)
                            ** (
                                2
                                * ((i % (rotary_size // rotary_ndims)) // 2)
                                / (rotary_size // rotary_ndims)
                            )
                        )
                    )
                    if i < rotary_size
                    else 1
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)

    naive_sin = np.array(
        [
            [
                [
                    math.sin(
                        position_ids[b, i // ((rotary_size) // rotary_ndims), m]
                        * (
                            (1 / base)
                            ** (
                                2
                                * ((i % (rotary_size // rotary_ndims)) // 2)
                                / (rotary_size // rotary_ndims)
                            )
                        )
                    )
                    if i < rotary_size
                    else 0
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)

    naive_cos[..., rotary_size:] = 1
    naive_sin[..., rotary_size:] = 0

    naive_x = x
    if x_layout == "BM(HK)" or x_layout == "BM(H2K)" or x_layout == "BM(H3K)":
        naive_x = x.reshape([B, M, H, -1, K])
    elif x_layout == "MB(HK)" or x_layout == "MB(H2K)" or x_layout == "MB(H3K)":
        naive_x = x.reshape([M, B, H, -1, K])

    naive_out = naive_embedding(
        naive_x,
        naive_cos,
        naive_sin,
        x_layout,
        B,
        M,
        H,
        K,
        dims,
        merged_dims,
        rotary_size,
        rotary_ndims,
        mode,
    )

    fused_cos = np.array(
        [
            [
                math.cos(
                    m
                    * (
                        (1 / base)
                        ** (
                            2
                            * ((i % (rotary_size // rotary_ndims)) // 2)
                            / (rotary_size // rotary_ndims)
                        )
                    )
                )
                for i in range(rotary_size // rotary_ndims)
            ]
            for m in range(2 * M)
        ]
    )
    fused_sin = np.array(
        [
            [
                math.sin(
                    m
                    * (
                        (1 / base)
                        ** (
                            2
                            * ((i % (rotary_size // rotary_ndims)) // 2)
                            / (rotary_size // rotary_ndims)
                        )
                    )
                )
                for i in range(rotary_size // rotary_ndims)
            ]
            for m in range(2 * M)
        ]
    )

    fused_x = flow.tensor(x, dtype=dtype, device=device)
    fused_cos = flow.tensor(fused_cos, dtype=dtype, device=device)
    fused_sin = flow.tensor(fused_sin, dtype=dtype, device=device)
    fused_position_ids = flow.tensor(position_ids, dtype=flow.int32, device=device)

    if x_layout == "BM(H3K)":
        out0 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=fused_cos,
            sin=fused_sin,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=0,
        )
        out1 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=fused_cos,
            sin=fused_sin,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=1,
        )
        out2 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=fused_cos,
            sin=fused_sin,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=2,
        )

        fused_out = np.concatenate((out0, out1, out2), axis=-1)
    else:
        fused_out = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=fused_cos,
            sin=fused_sin,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
        ).numpy()

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims),
            fused_out.reshape(merged_dims),
            atol=5e-2,
            rtol=5e-3,
        )
    )


def _test_with_position(
    test_case, x_layout, mode, base, rotary_size, dims, rotary_ndims, dtype, device
):
    B, M, H, K, merged_dims = parseDims(dims, x_layout)

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))

    position_ids = np.random.randint(2 * M, size=(B, rotary_ndims, M), dtype=int)

    naive_cos = np.array(
        [
            [
                [
                    math.cos(
                        position_ids[b, i // ((rotary_size) // rotary_ndims), m]
                        * (
                            (1 / base)
                            ** (
                                2
                                * ((i % (rotary_size / rotary_ndims)) // 2)
                                / (rotary_size / rotary_ndims)
                            )
                        )
                    )
                    if i < rotary_size
                    else 1
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)

    naive_sin = np.array(
        [
            [
                [
                    math.sin(
                        position_ids[b, i // ((rotary_size) // rotary_ndims), m]
                        * (
                            (1 / base)
                            ** (
                                2
                                * ((i % (rotary_size / rotary_ndims)) // 2)
                                / (rotary_size / rotary_ndims)
                            )
                        )
                    )
                    if i < rotary_size
                    else 0
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)

    naive_x = x
    if x_layout == "BM(HK)" or x_layout == "BM(H2K)" or x_layout == "BM(H3K)":
        naive_x = x.reshape([B, M, H, -1, K])
    elif x_layout == "MB(HK)" or x_layout == "MB(H2K)" or x_layout == "MB(H3K)":
        naive_x = x.reshape([M, B, H, -1, K])

    naive_out = naive_embedding(
        naive_x,
        naive_cos,
        naive_sin,
        x_layout,
        B,
        M,
        H,
        K,
        dims,
        merged_dims,
        rotary_size,
        rotary_ndims,
        mode,
    )

    fused_x = flow.tensor(x, dtype=dtype, device=device)
    fused_position_ids = flow.tensor(position_ids, dtype=flow.int32, device=device)

    if x_layout == "BM(H3K)":
        out0 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=0,
        )
        out1 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=1,
        )
        out2 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            output_layout="BMHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=2,
        )

        fused_out = np.concatenate((out0, out1, out2), axis=-1)
    else:
        fused_out = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
        ).numpy()

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims),
            fused_out.reshape(merged_dims),
            atol=5e-2,
            rtol=5e-3,
        )
    )


def _test_plane(
    test_case, x_layout, mode, base, rotary_size, dims, rotary_ndims, dtype, device
):
    B, M, H, K, merged_dims = parseDims(dims, x_layout)

    np.random.seed(3124)

    x = np.random.uniform(low=-1, high=1, size=(*merged_dims,))

    position_ids = np.random.randint(2 * M, size=(B, rotary_ndims, M), dtype=int)

    naive_cos = np.array(
        [
            [
                [
                    math.cos(
                        position_ids[b, i // ((rotary_size) // rotary_ndims), m]
                        * (
                            1
                            / (
                                base
                                ** (
                                    2
                                    * (i % (rotary_size // (2 * rotary_ndims)))
                                    / (rotary_size / rotary_ndims)
                                )
                            )
                        )
                    )
                    if i < rotary_size
                    else 1
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)

    naive_sin = np.array(
        [
            [
                [
                    math.sin(
                        position_ids[b, i // ((rotary_size) // rotary_ndims), m]
                        * (
                            1
                            / (
                                base
                                ** (
                                    2
                                    * (i % (rotary_size // (2 * rotary_ndims)))
                                    / (rotary_size / rotary_ndims)
                                )
                            )
                        )
                    )
                    if i < rotary_size
                    else 0
                    for i in range(K)
                ]
                for m in range(M)
            ]
            for b in range(B)
        ]
    ).reshape(B, 1, M, K)

    naive_x = x
    if x_layout == "BM(HK)" or x_layout == "BM(H2K)" or x_layout == "BM(H3K)":
        naive_x = x.reshape([B, M, H, -1, K])
    elif x_layout == "MB(HK)" or x_layout == "MB(H2K)" or x_layout == "MB(H3K)":
        naive_x = x.reshape([M, B, H, -1, K])

    naive_out = naive_embedding(
        naive_x,
        naive_cos,
        naive_sin,
        x_layout,
        B,
        M,
        H,
        K,
        dims,
        merged_dims,
        rotary_size,
        rotary_ndims,
        mode,
    )

    fused_x = flow.tensor(x, dtype=dtype, device=device)
    fused_position_ids = flow.tensor(position_ids, dtype=flow.int32, device=device)

    if x_layout == "MB(H3K)":
        out0 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            output_layout="MBHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=0,
        )
        out1 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            output_layout="MBHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=1,
        )
        out2 = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            output_layout="MBHK",
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
            tensor_index=2,
        )

        fused_out = np.concatenate((out0, out1, out2), axis=-1)
    else:
        fused_out = flow._C.fused_apply_rotary_emb(
            fused_x,
            cos=None,
            sin=None,
            position_ids=fused_position_ids,
            x_layout=x_layout,
            k_size=K,
            base=base,
            rotary_size=rotary_size,
            mode=mode,
        ).numpy()

    test_case.assertTrue(
        np.allclose(
            naive_out.reshape(merged_dims),
            fused_out.reshape(merged_dims),
            atol=5e-2,
            rtol=5e-3,
        )
    )


"""
1. if cos&sin is given, then base will not be used
2. if cos&sin is not given, then any form of x_layout which cannot infer the dimension of k is not allowed, e.g. BM(HK)
3. if position_ids is given, then M of cos&sin could be different from M of x
4. if position_ids is not given, the dimension of rotary positional embedding is by default 1
"""


@flow.unittest.skip_unless_1n1d()
class TestFusedRotaryEmbedding(flow.unittest.TestCase):
    # because rule no.2, kernels without cos&sin cannot work under specific x_layout
    def test_fused_rotary_embedding_op_plane(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_plane]
        args_dict["x_layout"] = ["MB(H3K)", "MB(HK)"]
        args_dict["mode"] = ["plane"]
        args_dict["base"] = [1e1]
        args_dict["rotary_size"] = [4, 8]
        args_dict["dims"] = [(3, 2, 5, 8)]
        args_dict["rotary_ndims"] = [2, 1]
        # args_dict["rotary_size"] = [48]
        # args_dict["dims"] = [(32, 2048, 32, 64)]
        args_dict["dtype"] = [flow.float16]
        args_dict["device"] = ["cuda"]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])

    def test_fused_rotary_embedding_op_interval_2d(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_with_position, _test_with_position_sinuous]
        args_dict["x_layout"] = ["BMHK"]
        args_dict["mode"] = ["interval"]
        args_dict["base"] = [1e1]
        args_dict["rotary_size"] = [4]
        args_dict["dims"] = [(3, 2, 5, 8)]
        args_dict["rotary_ndims"] = [2]
        # args_dict["rotary_size"] = [48]
        # args_dict["dims"] = [(32, 2048, 32, 64)]
        args_dict["dtype"] = [flow.float16]
        args_dict["device"] = ["cuda"]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])

    def test_fused_rotary_embedding_op_interval_1d(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [
            _test_without_position_sinuous,
            _test_without_position,
            _test_with_position,
            _test_with_position_sinuous,
        ]
        args_dict["x_layout"] = ["BMHK"]
        args_dict["mode"] = ["interval"]
        args_dict["base"] = [1e1]
        args_dict["rotary_size"] = [4]
        args_dict["dims"] = [(3, 2, 5, 8)]
        args_dict["rotary_ndims"] = [1]
        # args_dict["rotary_size"] = [48]
        # args_dict["dims"] = [(32, 2048, 32, 64)]
        args_dict["dtype"] = [flow.float16]
        args_dict["device"] = ["cuda"]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
