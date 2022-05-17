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

from oneflow.test_utils.test_util import GenArgDict

import oneflow as flow


def _np_tril(x, diagonal, fill_value, scale):
    if int(fill_value) == 0:
        return np.tril(x, diagonal) * scale

    upper = np.empty(x.shape)
    upper.fill(fill_value)
    upper = np.triu(upper, diagonal + 1)

    return np.tril(x, diagonal) * scale + upper


def _test_fused_scale_tril(
    test_case,
    shape,
    diagonal=0,
    fill_value=0,
    scale=1,
    dtype=flow.float32,
    device_type="cuda",
):
    if dtype is flow.int32 and not isinstance(scale, int):
        return

    if dtype is flow.int32:
        x = np.random.randint(0, 10, shape)
        y_grad = np.random.randint(0, 10, shape)
    else:
        x = np.random.rand(*shape)
        y_grad = np.random.rand(*shape)

    y = _np_tril(x, diagonal, fill_value, scale)
    x_grad = _np_tril(y_grad, diagonal, 0, scale)

    flow_x = flow.tensor(
        x, device=flow.device(device_type), dtype=dtype, requires_grad=True
    )
    flow_y = flow._C.fused_scale_tril(flow_x, diagonal, fill_value, scale)
    flow_y_grad = flow.tensor(y_grad, device=flow.device(device_type), dtype=dtype)
    flow_y.backward(flow_y_grad)

    flow_y_np = flow_y.numpy()
    test_case.assertTrue(np.allclose(flow_y_np, y.astype(flow_y_np.dtype)))

    flow_x_grad_np = flow_x.grad.numpy()
    test_case.assertTrue(
        np.allclose(flow_x_grad_np, x_grad.astype(flow_x_grad_np.dtype))
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class FusedScaleTrilTestCase(flow.unittest.TestCase):
    def test_fused_scale_tril(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(5, 5), (4, 6)]
        arg_dict["diagonal"] = [-1, 0, 1]
        arg_dict["fill_value"] = [-1, 0, 1]
        arg_dict["scale"] = [-2.3, 0.7, 2]
        arg_dict["dtype"] = [flow.float32]
        for kwargs in GenArgDict(arg_dict):
            _test_fused_scale_tril(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
