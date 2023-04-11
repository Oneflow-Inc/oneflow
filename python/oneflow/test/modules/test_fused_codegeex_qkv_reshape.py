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

import oneflow as flow
import oneflow.unittest


def _test_codegeex_qkv_reshape_impl(test_case, device, shape, num_attention_heads):
    query = flow.randn(shape).to("cuda")
    key = flow.randn(shape).to("cuda")
    value = flow.randn(shape).to("cuda")
    new_shape = (
        shape[0],
        shape[1],
        num_attention_heads,
        shape[2] / num_attention_heads,
    )
    new_query = query.view(new_shape)
    new_query = new_query.contiguous()
    new_key = key.view(new_shape)
    new_key = new_key.contiguous()
    new_value = value.view(new_shape)
    new_value = new_value.contiguous()
    (
        fused_new_query,
        fused_new_key,
        fused_new_value,
    ) = flow._C.fused_codegeex_qkv_reshape(query, key, value, num_attention_heads)

    def compare(a, b, rtol=1e-5, atol=1e-5):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=rtol, atol=atol
            ),
            f"\na\n{a.detach().cpu().numpy()}\n{'-' * 80}\nb:\n{b.detach().cpu().numpy()}\n{'*' * 80}\ndiff:\n{a.detach().cpu().numpy() - b.detach().cpu().numpy()}",
        )

    compare(new_query, fused_new_query)
    compare(new_key, fused_new_key)
    compare(new_value, fused_new_value)


@flow.unittest.skip_unless_1n1d()
class TestFusedCodegeexQkvReshapeModule(flow.unittest.TestCase):
    def test_codegeex_qkv_reshape(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_codegeex_qkv_reshape_impl]
        arg_dict["device"] = ["cuda"]
        arg_dict["shape"] = [(32, 8, 16), (32, 8, 32)]
        arg_dict["num_attention_heads"] = [(4), (8)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
