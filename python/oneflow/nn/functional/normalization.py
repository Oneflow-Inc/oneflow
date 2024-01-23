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
import oneflow as flow


def group_norm(
    input: flow.Tensor,
    num_groups: int,
    weight: flow.Tensor = None,
    bias: flow.Tensor = None,
    eps: float = 1e-05,
    num_channels: int = None,
):
    r"""Apply Group Normalization for last certain number of dimensions.

    See :class:`~oneflow.nn.GroupNorm` for details.
    """
    assert len(input.shape) >= 3, "The dimensions of input tensor must larger than 2"
    if num_channels is None:
        num_channels = input.shape[1]
    assert (
        input.shape[1] == num_channels
    ), "The channels of input tensor must equal num_channels"

    affine = weight is not None and bias is not None
    if input.is_cuda:
        return flow._C.group_norm(input, weight, bias, affine, num_groups, eps)
    else:
        origin_shape = input.shape
        reshape_to_1d = flow.reshape(input, shape=[origin_shape[0], num_groups, -1])
        mean = flow.mean(reshape_to_1d, dim=2, keepdim=True)
        variance = flow.var(reshape_to_1d, dim=2, unbiased=False, keepdim=True)
        normalized = (reshape_to_1d - mean) / flow.sqrt(variance + eps)
        normalized = flow.reshape(normalized, shape=[origin_shape[0], num_channels, -1])
        if weight is not None:
            normalized = normalized * weight.reshape(1, num_channels, 1)
        if bias is not None:
            normalized = normalized + bias.reshape(1, num_channels, 1)
        res = flow.reshape(normalized, shape=tuple(input.shape))
        return res
