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

import numpy as np
from oneflow.test_utils.test_util import GenArgList
import numpy as np

import oneflow as flow
import oneflow.unittest


def layernorm_ref(X, gamma, beta, normalized_shape, eps):
    X = X.float().cpu()
    gamma = gamma.float().cpu()
    beta = beta.float().cpu()
    feature_size = np.prod(normalized_shape)
    X_view = X.view(-1, feature_size)
    mean = X_view.mean(dim=-1, keepdim=True)
    var = X_view.var(dim=-1, unbiased=False, keepdim=True)
    Y = (X_view - mean) / flow.sqrt(var + eps)
    Y = Y * gamma.view(-1) + beta.view(-1)
    return Y.view(*X.size())


def _test_layernorm_forward(test_case, normalized_shape, device, dtype):
    layer_norm = flow.nn.LayerNorm(normalized_shape).to(device).to(dtype)
    X = flow.rand(2, *normalized_shape, dtype=dtype)
    Y_ref = layernorm_ref(
        X,
        layer_norm.weight.data,
        layer_norm.bias.data,
        normalized_shape,
        layer_norm.eps,
    )
    Y = layer_norm(X.to(device))
    test_case.assertTrue(np.allclose(Y.numpy(), Y_ref, 0.001, 0.001))


@flow.unittest.skip_unless_1n1d()
class TestlayernormCambriconModule(flow.unittest.TestCase):
    def test_layernorm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_layernorm_forward,
        ]
        arg_dict["normalized_shape"] = [
            [256, 256],
            [256, 256, 144],
            [512, 256],
            [512, 256, 144],
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [flow.float16, flow.float32]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
