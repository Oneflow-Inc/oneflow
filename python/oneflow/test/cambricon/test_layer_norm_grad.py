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
import numpy as np
import oneflow as flow


def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-6):
    begin_norm_axis = len(x.shape) - len(normalized_shape)
    begin_params_axis = len(x.shape) - len(normalized_shape)

    return flow._C.layer_norm_affine(
        x,
        weight,
        bias,
        begin_norm_axis=begin_norm_axis,
        begin_params_axis=begin_params_axis,
        epsilon=eps,
    )


def _get_stitching_grad(np_x, normalized_shape, np_weight=None, np_bias=None, eps=1e-6):
    x = flow.tensor(np_x).to("cpu").requires_grad_(True)
    weight = flow.tensor(np_weight).to("cpu").requires_grad_(True)

    feature_size = np.prod(normalized_shape)
    x_view = x.view(-1, feature_size)
    mean = x_view.mean(dim=-1, keepdim=True)
    var = x_view.var(dim=-1, unbiased=False, keepdim=True)
    lnorm = (x_view - mean) / flow.sqrt(var + eps)
    lnorm = lnorm.view(*x.size())

    dy = flow.ones(*lnorm.size(), dtype=flow.float32)
    d_lnorm = dy * weight
    d_bias = dy.sum(0)
    d_weight = (dy * lnorm).sum(0)

    n_in = np.prod(x.shape[1:])
    lnorm = lnorm.view(-1, n_in)
    d_lnorm = d_lnorm.view(lnorm.size())
    dx = (
        n_in * d_lnorm - d_lnorm.sum(1, True) - lnorm * (d_lnorm * lnorm).sum(1, True)
    ) / (n_in * flow.sqrt(var + eps))
    print(type(dx))
    dx = flow.tensor(dx)
    dx = dx.view(*x.size())

    return dx, d_weight, d_bias


def _get_mlu_grad(np_x, normalized_shape, np_weight, np_bias, eps=1e-6):
    x = flow.tensor(np_x).to("mlu").requires_grad_(True)
    weight = flow.tensor(np_weight).to("mlu").requires_grad_(True)
    bias = flow.tensor(np_bias).to("mlu").requires_grad_(True)
    _layer_norm(x, normalized_shape, weight, bias, eps).sum().backward()
    return x.grad, weight.grad, bias.grad


def _test_layer_norm_grad(normalized_shape):
    x_shape = (2,) + normalized_shape

    np_x = np.random.randn(*x_shape).astype(np.float32)
    np_weight = np.random.randn(*normalized_shape).astype(np.float32)
    np_bias = np.random.randn(*normalized_shape).astype(np.float32)

    mlu_dx, mlu_dweight, mlu_dbias = _get_mlu_grad(
        np_x, normalized_shape, np_weight, np_bias
    )
    stitching_dx, stitching_dweight, stitching_dbias = _get_stitching_grad(
        np_x, normalized_shape, np_weight, np_bias
    )
    assert np.allclose(mlu_dx, stitching_dx, 1e-4, 1e-4)
    assert np.allclose(mlu_dweight, stitching_dweight, 1e-4, 1e-4)
    assert np.allclose(mlu_dbias, stitching_dbias, 1e-4, 1e-4)


def test_layer_norm_grad():
    for normalized_shape in [(256, 256), (256, 256, 144), (512, 256), (512, 256, 144)]:
        _test_layer_norm_grad(normalized_shape)
