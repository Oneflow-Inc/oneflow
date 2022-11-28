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
import os
import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow


def _test_fused_cross_feature_interaction_v1(
    test_case, batchsize, in_feature, dtype, device,
):
    x = np.random.uniform(low=-1, high=1, size=(batchsize, in_feature))
    weight = np.random.uniform(low=-1, high=1, size=(1, in_feature))
    bias = np.random.uniform(low=-1, high=1, size=(in_feature))
    x0 = np.random.uniform(low=-1, high=1, size=(batchsize, in_feature))

    fused_x = flow.tensor(x, dtype=dtype, device=device, requires_grad=True)
    naive_x = flow.tensor(x, dtype=dtype, device=device, requires_grad=True)
    fused_weight = flow.tensor(weight, dtype=dtype, device=device, requires_grad=True)
    naive_weight = flow.tensor(weight, dtype=dtype, device=device, requires_grad=True)
    fused_bias = flow.tensor(bias, dtype=dtype, device=device, requires_grad=True)
    naive_bias = flow.tensor(bias, dtype=dtype, device=device, requires_grad=True)
    fused_x0 = flow.tensor(x0, dtype=dtype, device=device, requires_grad=True)
    naive_x0 = flow.tensor(x0, dtype=dtype, device=device, requires_grad=True)

    fused_out = flow._C.fused_cross_feature_interaction(
        fused_x, fused_weight, fused_x0, fused_bias, "vector"
    )

    naive_out = (
        flow._C.matmul(naive_x, naive_weight, transpose_b=True) * naive_x0 + naive_bias
    ) + naive_x

    total_out = fused_out.sum() + naive_out.sum()
    total_out.backward()

    test_case.assertTrue(
        np.allclose(fused_out.numpy(), naive_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    test_case.assertTrue(
        np.allclose(fused_x.grad.numpy(), naive_x.grad.numpy(), atol=1e-4, rtol=1e-4,)
    )
    test_case.assertTrue(
        np.allclose(
            fused_weight.grad.numpy(), naive_weight.grad.numpy(), atol=1e-4, rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(fused_x0.grad.numpy(), naive_x0.grad.numpy(), atol=1e-4, rtol=1e-4,)
    )
    test_case.assertTrue(
        np.allclose(
            fused_bias.grad.numpy(), naive_bias.grad.numpy(), atol=1e-4, rtol=1e-4,
        )
    )


def _test_fused_cross_feature_interaction_v2(
    test_case, batchsize, in_feature, dtype, device,
):
    x = np.random.uniform(low=-1, high=1, size=(batchsize, in_feature))
    weight = np.random.uniform(low=-1, high=1, size=(in_feature, in_feature))
    bias = np.random.uniform(low=-1, high=1, size=(in_feature))
    x0 = np.random.uniform(low=-1, high=1, size=(batchsize, in_feature))

    fused_x = flow.tensor(x, dtype=dtype, device=device, requires_grad=True)
    naive_x = flow.tensor(x, dtype=dtype, device=device, requires_grad=True)
    fused_weight = flow.tensor(weight, dtype=dtype, device=device, requires_grad=True)
    naive_weight = flow.tensor(weight, dtype=dtype, device=device, requires_grad=True)
    fused_bias = flow.tensor(bias, dtype=dtype, device=device, requires_grad=True)
    naive_bias = flow.tensor(bias, dtype=dtype, device=device, requires_grad=True)
    fused_x0 = flow.tensor(x0, dtype=dtype, device=device, requires_grad=True)
    naive_x0 = flow.tensor(x0, dtype=dtype, device=device, requires_grad=True)

    fused_out = flow._C.fused_cross_feature_interaction(
        fused_x, fused_weight, fused_x0, fused_bias, "matrix"
    )

    naive_out = (
        flow._C.bias_add(
            flow._C.matmul(naive_x, naive_weight, transpose_b=True), naive_bias, axis=1
        )
        * naive_x0
        + naive_x
    )

    total_out = fused_out.sum() + naive_out.sum()
    total_out.backward()

    test_case.assertTrue(
        np.allclose(fused_out.numpy(), naive_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    test_case.assertTrue(
        np.allclose(fused_x.grad.numpy(), naive_x.grad.numpy(), atol=1e-4, rtol=1e-4,)
    )
    test_case.assertTrue(
        np.allclose(
            fused_weight.grad.numpy(), naive_weight.grad.numpy(), atol=1e-4, rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(fused_x0.grad.numpy(), naive_x0.grad.numpy(), atol=1e-4, rtol=1e-4,)
    )
    test_case.assertTrue(
        np.allclose(
            fused_bias.grad.numpy(), naive_bias.grad.numpy(), atol=1e-4, rtol=1e-4,
        )
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestFusedCrossFeatureInteraction(flow.unittest.TestCase):
    def test_fused_cross_feature_interaction_v1(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_fused_cross_feature_interaction_v1]
        args_dict["batchsize"] = [1, 2, 4]
        args_dict["in_feature"] = [32, 64, 96, 128]
        args_dict["dtype"] = [flow.float32]
        args_dict["device"] = ["cuda"]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])

    def test_fused_cross_feature_interaction_v2(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_fused_cross_feature_interaction_v2]
        args_dict["batchsize"] = [1, 2, 4]
        args_dict["in_feature"] = [32, 64, 96, 128]
        args_dict["dtype"] = [flow.float32]
        args_dict["device"] = ["cuda"]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
