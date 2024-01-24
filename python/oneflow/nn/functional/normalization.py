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


def layer_norm(input, normalized_shape: tuple, weight=None, bias=None, eps=1e-05):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    normalized_shape = tuple(normalized_shape)
    assert len(input.shape) > len(
        normalized_shape
    ), "Input tensor dim must greater than normalized dim!"
    begin_norm_axis = len(input.shape) - len(normalized_shape)
    begin_params_axis = len(input.shape) - len(normalized_shape)

    elementwise_affine = True if (weight is not None and bias is not None) else False

    for i in range(0, len(normalized_shape)):
        if input.shape[i + begin_params_axis] != normalized_shape[i]:
            raise RuntimeError(
                f"Given normalized_shape={normalized_shape}, expected input with shape [*, {str(normalized_shape)[1:-1]}], but got input of size {input.shape}"
            )

    if input.is_cpu:
        reduce_axis = []
        for dim in range(len(input.shape)):
            if dim >= begin_norm_axis:
                reduce_axis.append(dim)
        mean = input.mean(dim=reduce_axis, keepdim=True)
        variance = input.var(dim=reduce_axis, unbiased=False, keepdim=True)
        params_shape = input.shape[begin_params_axis:]
        if len(mean.shape) == 1:
            nd_params_shape = [1] * len(input.shape)
            nd_params_shape[begin_norm_axis] = params_shape[0]
            mean = flow.reshape(mean, shape=nd_params_shape)
            variance = flow.reshape(variance, nd_params_shape)
            if weight is not None and params_shape[0] == weight.nelement():
                weight = flow.reshape(weight, shape=nd_params_shape)
            if bias is not None and params_shape[0] == bias.nelement():
                bias = flow.reshape(bias, shape=nd_params_shape)
        elif len(mean.shape) == len(input.shape):
            pass
        else:
            raise ValueError(
                "shape of mean and variance should be 1D or has number of axes and x's"
            )
        variance += eps
        normalized = (input - mean) * variance.rsqrt()
        if elementwise_affine:
            normalized = normalized * weight + bias
        return normalized
    else:
        if elementwise_affine:
            res = flow._C.layer_norm_affine(
                input,
                weight,
                bias,
                begin_norm_axis=begin_norm_axis,
                begin_params_axis=begin_params_axis,
                epsilon=eps,
            )
        else:
            res = flow._C.layer_norm(
                input,
                begin_norm_axis=begin_norm_axis,
                begin_params_axis=begin_params_axis,
                epsilon=eps,
            )
        return res
