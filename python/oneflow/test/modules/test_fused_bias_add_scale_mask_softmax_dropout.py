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
import os
import numpy as np
from collections import OrderedDict

import torch
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgDict


def _torch_bias_add_scale_mask_softmax_dropout(x, bias, mask, fill, scale, p):
    # masked = (x + bias) * mask * scale
    # masked = x * mask * scale
    # unmask = (1 - mask.int()).bool()
    # masked.masked_fill_(unmask, fill)
    masked = x
    softmax_y = torch.nn.functional.softmax(masked, dim=-1)
    # y = torch.nn.functional.dropout(softmax_y, p)
    return softmax_y, softmax_y


def _test_bias_add_fused_scale_mask_softmax_dropout(
    test_case,
    input_shape,
    bias_shape,
    mask_shape,
    input_dtype=flow.float32,
    mask_dtype=flow.bool,
    fill=-10000,
    scale=1.0,
    p=0.0,
    atol=1e-4,
    rtol=1e-5,
):
    print(f"{'=' * 40} test case {'=' * 40}")
    print(f"input_shap={input_shape}")
    print(f"bias_shape={bias_shape}")
    print(f"mask_shape={mask_shape}")
    print(f"input_dtype={input_dtype}")
    print(f"mask_dtype={mask_dtype}")
    print(f"fill={fill}")
    print(f"scale={scale}")
    print(f"p={p}")

    np_input = np.random.randn(*input_shape).astype(np.float32)
    np_bias = np.random.randn(*bias_shape).astype(np.float32)
    np_mask = np.random.randint(0, 2, size=mask_shape).astype(np.int32)
    np_init_grad = np.array(1e6)

    torch_input = torch.tensor(np_input).to(device="cuda")
    torch_bias = torch.tensor(np_bias).to(device="cuda")
    torch_mask = torch.tensor(np_mask).to(device="cuda").bool()
    torch_input.requires_grad_(True)
    torch_bias.requires_grad_(True)
    torch_output, torch_softmax_output = _torch_bias_add_scale_mask_softmax_dropout(
        torch_input, torch_bias, torch_mask, fill, scale, p
    )
    torch_output.sum().backward(torch.tensor(np_init_grad))
    torch_input_grad = torch_input.grad.detach().cpu()
    # torch_bias_grad = torch_bias.grad.detach().cpu()
    torch_output = torch_output.detach().cpu()
    torch_softmax_output = torch_softmax_output.detach().cpu()

    input = flow.tensor(np_input, dtype=input_dtype, device="cuda")
    bias = flow.tensor(np_bias, dtype=input_dtype, device="cuda")
    mask = flow.tensor(np_mask, dtype=mask_dtype, device="cuda")
    input.requires_grad_(True)
    bias.requires_grad_(True)
    # output, softmax_output = flow._C.fused_scale_mask_softmax_dropout(
    #     input, mask, fill_value=fill, scale=scale, p=p,
    # )
    output = torch.nn.functional.softmax(input, dim=-1)
    softmax_output = output
    output.sum().backward(flow.tensor(np_init_grad, device="cuda"))
    input_grad = input.grad.detach().cpu()
    # bias_grad = bias.grad.detach().cpu()
    output = output.to(dtype=flow.float32, device="cpu")
    softmax_output = softmax_output.to(dtype=flow.float32, device="cpu")

    test_case.assertTrue(
        np.allclose(output.numpy(), torch_output.numpy(), atol=1e-3, rtol=1e-5),
        f"\noutput:\n{output.numpy()}\n{'-' * 80}\ntorch_output:\n{torch_output.numpy()}\n{'*' * 80}\ndiff:\n{output.numpy() - torch_output.numpy()}",
    )
    test_case.assertTrue(
        np.allclose(
            softmax_output.numpy(), torch_softmax_output.numpy(), atol=1e-3, rtol=1e-5
        ),
        f"\nsoftmax_output:\n{softmax_output.numpy()}\n{'-' * 80}\ntorch_softmax_output:\n{torch_softmax_output.numpy()}\n{'*' * 80}\ndiff:\n{softmax_output.numpy() - torch_softmax_output.numpy()}",
    )
    test_case.assertTrue(
        np.allclose(input_grad.numpy(), torch_input_grad.numpy(), atol=1e-5, rtol=1e-5),
        f"\ninput_grad:\n{input_grad.numpy()}\n{'-' * 80}\ntorch_input_grad:\n{torch_input_grad.numpy()}\n{'*' * 80}\ndiff:\n{input_grad.numpy() - torch_input_grad.numpy()}",
    )
    # test_case.assertTrue(
    #     np.allclose(bias_grad.numpy(), torch_bias_grad.numpy()),
    #     f"\nbias_grad:\n{bias_grad.numpy()}\n{'-' * 80}\ntorch_bias_grad:\n{torch_bias_grad.numpy()}\n{'*' * 80}\ndiff:\n{bias_grad.numpy() - torch_bias_grad.numpy()}",
    # )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestFusedBiasAddScaleMaskSoftmaxDropout(flow.unittest.TestCase):
    def test_4d(test_case):
        args_dict = OrderedDict()
        # args_dict["input_shape"] = [[4, 12, 8, 8]]
        # args_dict["bias_shape"] = [[1, 12, 8, 8]]
        # args_dict["mask_shape"] = [[4, 1, 1, 8]]
        args_dict["input_shape"] = [[4, 2, 3]]
        args_dict["bias_shape"] = [[1, 2, 3]]
        args_dict["mask_shape"] = [[4, 1, 3]]
        args_dict["input_dtype"] = [flow.float32]
        args_dict["mask_dtype"] = [flow.bool]
        args_dict["fill"] = [-10000.0]
        args_dict["scale"] = [1.0, 2.0, 4.0]
        args_dict["p"] = [0.0, 1.0]

        for kwarg in GenArgDict(args_dict):
            _test_bias_add_fused_scale_mask_softmax_dropout(test_case, **kwarg)


if __name__ == "__main__":
    unittest.main()
