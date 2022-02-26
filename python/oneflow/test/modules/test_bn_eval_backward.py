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
from test_util import GenArgDict
import numpy as np
import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest
import os


def _test_bn_eval_backward(test_case, axis, dtype):
    n = 16
    c = 64
    h = 112
    w = 112
    epsilon = 1e-5
    if dtype == flow.float16:
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    if axis == 1:
        in_shape = (n, c, h, w)
    else:
        in_shape = (n, h, w, c)
    broadcast_shape = [1, 1, 1, 1]
    broadcast_shape[axis] = c
    np_dy = np.random.rand(*in_shape).astype(np_dtype)
    np_gamma = np.random.rand(c).astype(np.float32)
    np_variance = np.random.rand(c).astype(np.float32)
    dy_tensor = flow.tensor(np_dy, device="cuda", requires_grad=False)
    gamma_tensor = flow.tensor(np_gamma, device="cuda", requires_grad=False)
    variance_tensor = flow.tensor(np_variance, device="cuda", requires_grad=False)
    # dx bn_eval_backward
    dx = flow._C.bn_eval_backward(
        dy_tensor, gamma_tensor, variance_tensor, axis, epsilon
    )

    # dx origin
    scalar_add = variance_tensor + epsilon
    var_rsqrt = flow.rsqrt(scalar_add)
    reshape_gamma = flow.reshape(gamma_tensor, broadcast_shape)
    reshape_inv_var = flow.reshape(var_rsqrt, broadcast_shape)

    dx_origin = flow.cast(
        flow.cast(dy_tensor, flow.float) * reshape_gamma * reshape_inv_var,
        dy_tensor.dtype,
    )

    test_case.assertTrue(
        np.allclose(dx.numpy(), dx_origin.numpy(), rtol=1e-4, atol=1e-4,)
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class BnEvalTestCase(flow.unittest.TestCase):
    def test_bn_eval_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["axis"] = [1, 3]
        arg_dict["dtype"] = [flow.float16, flow.float32]
        for kwargs in GenArgDict(arg_dict):
            _test_bn_eval_backward(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
