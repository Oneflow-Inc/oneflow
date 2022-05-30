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


def _matmul_bias_relu(x, weight, bias, skip_activate):
    # We do not add dropout in unittest, cause its result is random.
    out = flow._C.bias_add(flow._C.matmul(x, weight, transpose_b=True), bias, axis=1)
    if not skip_activate:
        out = flow._C.relu(out)
    return out


def _test_fused_matmul_bias_add_relu_dropout(
    test_case,
    batchsize,
    in_feature,
    hidden_size_list,
    out_feature,
    skip_final_activation,
    dtype,
    device,
):
    x = np.random.uniform(low=-1, high=1, size=(batchsize, in_feature))

    fused_x = flow.tensor(x, dtype=dtype, device=device, requires_grad=True)
    naive_x = flow.tensor(x, dtype=dtype, device=device, requires_grad=True)

    fused_weight_list = []
    naive_weight_list = []
    fused_bias_list = []
    naive_bias_list = []

    hidden_num = len(hidden_size_list)

    if hidden_num != 0:
        np_first_weight = np.random.uniform(
            low=-1, high=1, size=(hidden_size_list[0], in_feature)
        )
        np_first_bias = np.random.uniform(low=-1, high=1, size=hidden_size_list[0])

        fused_weight_list.append(
            flow.tensor(np_first_weight, dtype=dtype, device=device, requires_grad=True)
        )
        fused_bias_list.append(
            flow.tensor(np_first_bias, dtype=dtype, device=device, requires_grad=True)
        )
        naive_weight_list.append(
            flow.tensor(np_first_weight, dtype=dtype, device=device, requires_grad=True)
        )
        naive_bias_list.append(
            flow.tensor(np_first_bias, dtype=dtype, device=device, requires_grad=True)
        )

    for idx in range(1, hidden_num):
        np_weight = np.random.uniform(
            low=-1, high=1, size=(hidden_size_list[idx], hidden_size_list[idx - 1])
        )
        np_bias = np.random.uniform(low=-1, high=1, size=hidden_size_list[idx])

        fused_weight_list.append(
            flow.tensor(np_weight, dtype=dtype, device=device, requires_grad=True)
        )
        fused_bias_list.append(
            flow.tensor(np_bias, dtype=dtype, device=device, requires_grad=True)
        )
        naive_weight_list.append(
            flow.tensor(np_weight, dtype=dtype, device=device, requires_grad=True)
        )
        naive_bias_list.append(
            flow.tensor(np_bias, dtype=dtype, device=device, requires_grad=True)
        )

    np_final_weight = np.random.uniform(low=-1, high=1, size=(out_feature, in_feature))

    if hidden_num != 0:
        np_final_weight = np.random.uniform(
            low=-1, high=1, size=(out_feature, hidden_size_list[-1])
        )

    np_final_bias = np.random.uniform(low=-1, high=1, size=(out_feature))

    fused_weight_list.append(
        flow.tensor(np_final_weight, dtype=dtype, device=device, requires_grad=True)
    )
    fused_bias_list.append(
        flow.tensor(np_final_bias, dtype=dtype, device=device, requires_grad=True)
    )
    naive_weight_list.append(
        flow.tensor(np_final_weight, dtype=dtype, device=device, requires_grad=True)
    )
    naive_bias_list.append(
        flow.tensor(np_final_bias, dtype=dtype, device=device, requires_grad=True)
    )

    fused_out = flow._C.fused_matmul_bias_add_relu_dropout(
        fused_x,
        fused_weight_list,
        fused_bias_list,
        # We do not add dropout in unittest, cause its result is random.
        dropout_rate_list=[0.0] * len(fused_weight_list),
        skip_final_activation=skip_final_activation,
    )

    naive_out = _matmul_bias_relu(
        naive_x,
        naive_weight_list[0],
        naive_bias_list[0],
        False if hidden_num != 0 else skip_final_activation,
    )

    for idx in range(1, hidden_num + 1):
        if idx == hidden_num:
            naive_out = _matmul_bias_relu(
                naive_out,
                naive_weight_list[idx],
                naive_bias_list[idx],
                skip_final_activation,
            )
        else:
            naive_out = _matmul_bias_relu(
                naive_out, naive_weight_list[idx], naive_bias_list[idx], False
            )

    total_out = fused_out.sum() + naive_out.sum()
    total_out.backward()

    test_case.assertTrue(
        np.allclose(fused_out.numpy(), naive_out.numpy(), atol=1e-4, rtol=1e-4)
    )

    # Test weight grad equality
    for idx in range(hidden_num + 1):
        test_case.assertTrue(
            np.allclose(
                fused_weight_list[idx].grad.numpy(),
                naive_weight_list[idx].grad.numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )
        test_case.assertTrue(
            np.allclose(
                fused_bias_list[idx].grad.numpy(),
                naive_bias_list[idx].grad.numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )
    # Test dx equality
    test_case.assertTrue(
        np.allclose(fused_x.grad.numpy(), naive_x.grad.numpy(), atol=1e-4, rtol=1e-4)
    )


@flow.unittest.skip_unless_1n1d()
class TestFusedMatmulBiasAddReluDropout(flow.unittest.TestCase):
    def test_fused_matmul_bias_add_relu_dropout(test_case):
        args_dict = OrderedDict()
        args_dict["test_func"] = [_test_fused_matmul_bias_add_relu_dropout]
        args_dict["batchsize"] = [1, 2, 4]
        args_dict["in_feature"] = [96, 128, 64]
        args_dict["hidden_size_list"] = [[256, 512], [400, 400, 400, 400], [17, 33, 79]]
        args_dict["out_feature"] = [512, 400, 1024, 1]
        args_dict["skip_final_activation"] = [False]
        args_dict["dtype"] = [flow.float32]
        args_dict["device"] = ["cuda"]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
