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


def _matmul_bias(x, weight, bias, add_to_output):
    return flow._C.add(
        flow._C.bias_add(
            flow._C.matmul(x, weight, transpose_b=True), bias, axis=len(x.shape) - 1
        ),
        add_to_output,
    )


def _test_fused_matmul_add_bias(
    test_case, batchsize, in_feature, out_feature, _add_to_output, dtype, device,
):
    add_to_output = np.zeros((*batchsize, out_feature))
    if _add_to_output:
        add_to_output = np.random.uniform(
            low=-1, high=1, size=(*batchsize, out_feature)
        )
    x = np.random.uniform(low=-1, high=1, size=(*batchsize, in_feature))
    weight = np.random.uniform(low=-1, high=1, size=(out_feature, in_feature))
    bias = np.random.uniform(low=-1, high=1, size=(out_feature))

    naive_x = flow.tensor(x, dtype=dtype, requires_grad=True)
    naive_weight = flow.tensor(weight, dtype=dtype, requires_grad=True)
    naive_bias = flow.tensor(bias, dtype=dtype, requires_grad=True)
    naive_add_to_output = flow.tensor(add_to_output, dtype=dtype, requires_grad=True)

    fused_x = flow.tensor(x, dtype=dtype, device=device, requires_grad=True)
    fused_weight = flow.tensor(weight, dtype=dtype, device=device, requires_grad=True)
    fused_bias = flow.tensor(bias, dtype=dtype, device=device, requires_grad=True)
    fused_add_to_output = None
    if _add_to_output:
        fused_add_to_output = flow.tensor(
            add_to_output, dtype=dtype, device=device, requires_grad=False
        )

    navie_y = _matmul_bias(naive_x, naive_weight, naive_bias, naive_add_to_output)
    fused_y = flow._C.fused_matmul_bias(
        fused_x, fused_weight, fused_bias, fused_add_to_output
    )

    y = navie_y.sum() + fused_y.sum()
    y.backward()

    # TODO: relative error might be too high...
    # Test output equality
    if _add_to_output:
        test_case.assertTrue(
            np.allclose(navie_y.numpy(), fused_y.numpy(), atol=5e-2, rtol=1e-4)
        )
    else:
        test_case.assertTrue(
            np.allclose(navie_y.numpy(), fused_y.numpy(), atol=5e-2, rtol=1e-4)
        )

    # Test grad equality
    test_case.assertTrue(
        np.allclose(naive_x.grad.numpy(), fused_x.grad.numpy(), atol=5e-2, rtol=1e-4)
    )

    test_case.assertTrue(
        np.allclose(
            naive_weight.grad.numpy(), fused_weight.grad.numpy(), atol=5e-2, rtol=1e-4
        )
    )
    test_case.assertTrue(
        np.allclose(
            naive_bias.grad.numpy(), fused_bias.grad.numpy(), atol=1e-4, rtol=1e-4
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestFusedMatmulBiasAddRelu(flow.unittest.TestCase):
    def test_fused_matmul_op(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_fused_matmul_add_bias]
        args_dict["batchsize"] = [
            (1,),
            (4,),
            (8,),
            (2, 4),
            (2, 4, 8),
            (2, 4, 4, 4, 8),
        ]
        args_dict["in_feature"] = [96, 128]
        args_dict["out_feature"] = [512, 1024, 288, 1]
        args_dict["_add_to_output"] = [True]
        args_dict["dtype"] = [flow.float32, flow.float64]
        args_dict["device"] = ["cuda", "cpu"]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
