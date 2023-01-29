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
import math

import numpy as np
import oneflow as flow
from oneflow import nn

import unittest
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgList


def get_timestep_embedding(
    timesteps: flow.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    N = 1

    origin_shape = timesteps.shape

    for i in range(len(timesteps.shape)):
        N = N * timesteps.shape[i]

    timesteps = timesteps.reshape((N, -1)).squeeze()

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * flow.arange(
        start=0, end=half_dim, dtype=flow.float32
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = flow.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = flow.cat([flow.sin(emb), flow.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = flow.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = flow.nn.functional.pad(emb, (0, 1, 0, 0))

    emb = emb.reshape((*origin_shape, embedding_dim)).squeeze()
    return emb


def get_interleaved_timestep_embedding(
    timesteps: flow.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    N = 1

    origin_shape = timesteps.shape

    for i in range(len(timesteps.shape)):
        N = N * timesteps.shape[i]

    timesteps = timesteps.reshape((N, -1)).squeeze()

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * flow.arange(
        start=0, end=half_dim, dtype=flow.float32
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = flow.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb0 = flow.sin(emb[:, 0])
    emb0 = emb0.reshape([len(timesteps), -1])
    for i in range(1, embedding_dim - embedding_dim % 2):
        if i % 2 == 1:
            emb0 = flow.cat(
                [emb0, flow.cos(emb[:, i // 2]).reshape([len(timesteps), -1])], dim=-1
            )
        else:
            emb0 = flow.cat(
                [emb0, flow.sin(emb[:, i // 2]).reshape([len(timesteps), -1])], dim=-1
            )

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb0 = flow.cos(emb[:, 0])
        emb0 = emb0.reshape([len(timesteps), -1])
        for i in range(1, embedding_dim - embedding_dim % 2):
            if i % 2 == 1:
                emb0 = flow.cat(
                    [emb0, flow.sin(emb[:, i // 2]).reshape([len(timesteps), -1])],
                    dim=-1,
                )
            else:
                emb0 = flow.cat(
                    [emb0, flow.cos(emb[:, i // 2]).reshape([len(timesteps), -1])],
                    dim=-1,
                )

    emb0 = emb0.reshape([-1, embedding_dim - embedding_dim % 2])
    # zero pad
    if embedding_dim % 2 == 1:
        emb0 = flow.nn.functional.pad(emb0, (0, 1, 0, 0))

    emb0 = emb0.reshape((*origin_shape, embedding_dim)).squeeze()
    return emb0


def navie_embedding(
    timesteps: flow.Tensor,
    embedding_dim: int,
    layout: int = 0,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    if layout == 0:
        return get_timestep_embedding(
            timesteps, embedding_dim, False, downscale_freq_shift, scale, max_period
        )
    elif layout == 1:
        return get_timestep_embedding(
            timesteps, embedding_dim, True, downscale_freq_shift, scale, max_period
        )
    elif layout == 2:
        return get_interleaved_timestep_embedding(
            timesteps, embedding_dim, False, downscale_freq_shift, scale, max_period
        )
    else:
        return get_interleaved_timestep_embedding(
            timesteps, embedding_dim, True, downscale_freq_shift, scale, max_period
        )


def _test_fused_sinusoidal_positional_encode(
    test_case,
    src_type,
    length,
    embedding_dim,
    layout,
    downscale_freq_shift,
    scale,
    max_period,
):
    if src_type == np.int32:
        positions = flow.tensor(
            np.random.randint(low=1024, size=length, dtype=src_type),
            dtype=flow.int32,
            device="cuda",
            requires_grad=False,
        )
    elif src_type == np.float32:
        positions = flow.tensor(
            np.random.rand(*length),
            dtype=flow.float32,
            device="cuda",
            requires_grad=False,
        )

    fused = flow._C.fused_sinusoidal_positional_encode(
        positions, embedding_dim, layout, downscale_freq_shift, scale, max_period
    )
    naive = navie_embedding(
        positions, embedding_dim, layout, downscale_freq_shift, scale, max_period
    )

    test_case.assertTrue(
        np.allclose(naive.numpy(), fused.numpy(), atol=1e-2, rtol=1e-2)
    )


@flow.unittest.skip_unless_1n1d()
class TestFusedSinusoidalPositionalEncode(flow.unittest.TestCase):
    def test_fused_matmul_op(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_fused_sinusoidal_positional_encode]
        args_dict["src_type"] = [np.int32, np.float32]
        args_dict["length"] = [(7,), (512,), (5, 64), (2, 3, 4, 5)]
        args_dict["embedding_dim"] = [17, 32, 512]
        args_dict["layout"] = [0, 1, 2, 3]
        args_dict["downscale_freq_shift"] = [0.3]

        args_dict["scale"] = [3.3]
        args_dict["max_period"] = [27, 10000]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
