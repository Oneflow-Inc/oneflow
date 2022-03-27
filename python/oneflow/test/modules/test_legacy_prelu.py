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
import unittest
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgList


def legacy_prelu_channel_first_test(test_case, shape, dtype, device):
    channel_num = shape[1]
    shape_length = len(shape)
    np_x = np.random.uniform(low=-1, high=1, size=shape).astype(np.float32)
    np_prelu_weight = np.random.uniform(low=-1, high=1, size=(channel_num,)).astype(
        np.float32
    )
    np_legacy_prelu_weight = np.reshape(
        np_prelu_weight, [channel_num if i == 0 else 1 for i in range(shape_length - 1)]
    )

    x_tensor = flow.tensor(np_x, requires_grad=True, dtype=dtype, device=device)
    w_tensor = flow.tensor(
        np_prelu_weight, requires_grad=True, dtype=dtype, device=device
    )

    legacy_x_tensor = flow.tensor(np_x, requires_grad=True, dtype=dtype, device=device)
    legacy_w_tensor = flow.tensor(
        np_legacy_prelu_weight, requires_grad=True, dtype=dtype, device=device
    )

    origin_out = flow._C.prelu(x_tensor, w_tensor)
    legacy_out = flow._C.prelu(legacy_x_tensor, legacy_w_tensor)

    test_case.assertTrue(
        np.allclose(origin_out.numpy(), legacy_out.numpy(), rtol=1e-4, atol=1e-4,)
    )

    loss = (origin_out + legacy_out).sum()
    loss.backward()

    test_case.assertTrue(
        np.allclose(
            x_tensor.grad.numpy(), legacy_x_tensor.grad.numpy(), rtol=1e-4, atol=1e-4,
        )
    )

    test_case.assertTrue(
        np.allclose(
            w_tensor.grad.numpy(),
            np.reshape(legacy_w_tensor.grad.numpy(), (channel_num,)),
            rtol=1e-4,
            atol=1e-4,
        )
    )


def legacy_prelu_channel_last_test(test_case, shape, dtype, device):
    channel_num = shape[1]
    shape_length = len(shape)
    np_x = np.random.uniform(low=-1, high=1, size=shape).astype(np.float32)
    np_prelu_weight = np.random.uniform(low=-1, high=1, size=(channel_num,)).astype(
        np.float32
    )
    np_legacy_prelu_weight = np.reshape(
        np_prelu_weight, [channel_num if i == 0 else 1 for i in range(shape_length - 1)]
    )

    x_tensor = flow.tensor(np_x, requires_grad=True, dtype=dtype, device=device)
    w_tensor = flow.tensor(
        np_prelu_weight, requires_grad=True, dtype=dtype, device=device
    )

    origin_out = flow._C.prelu(x_tensor, w_tensor)
    permute_axis = [0 if i == 1 else i for i in range(1, shape_length)]
    permute_axis.append(1)
    permuted_origin_out = flow.permute(origin_out, permute_axis)

    legacy_x_tensor = flow.tensor(np_x, requires_grad=True, dtype=dtype, device=device)
    legacy_w_tensor = flow.tensor(
        np_legacy_prelu_weight, requires_grad=True, dtype=dtype, device=device
    )
    nhwc_legacy_x_tensor = flow.permute(legacy_x_tensor, permute_axis)
    nhwc_legacy_w_tensor = flow.reshape(
        legacy_w_tensor,
        [channel_num if i == shape_length - 2 else 1 for i in range(shape_length - 1)],
    )

    legacy_out = flow._C.prelu(nhwc_legacy_x_tensor, nhwc_legacy_w_tensor)

    test_case.assertTrue(
        np.allclose(
            permuted_origin_out.numpy(), legacy_out.numpy(), rtol=1e-4, atol=1e-4,
        )
    )

    loss = (permuted_origin_out + legacy_out).sum()
    loss.backward()

    test_case.assertTrue(
        np.allclose(
            x_tensor.grad.numpy(), legacy_x_tensor.grad.numpy(), rtol=1e-4, atol=1e-4,
        )
    )

    test_case.assertTrue(
        np.allclose(
            w_tensor.grad.numpy(),
            np.reshape(legacy_w_tensor.grad.numpy(), (channel_num,)),
            rtol=1e-4,
            atol=1e-4,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestLegacyPrelu(flow.unittest.TestCase):
    def test_channel_first_legacy_prelu(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            legacy_prelu_channel_first_test,
            legacy_prelu_channel_last_test,
        ]
        arg_dict["shape"] = [
            [4, 63, 32, 32],
            [2, 32, 64],
            [6, 16, 4, 4, 8],
            [1, 128, 32, 64],
        ]
        arg_dict["dtype"] = [flow.float32, flow.float64]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
