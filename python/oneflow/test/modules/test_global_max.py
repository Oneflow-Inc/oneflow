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
import numpy as np
import oneflow as flow
import oneflow.unittest
from collections import OrderedDict
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList


def _np_max(shape, dim, keepdims):
    # np array result
    input_arr = np.random.randn(*shape)
    np_out = np.amax(input_arr, axis=dim, keepdims=keepdims)
    np_out_grad = np.zeros_like(input_arr)
    if dim == None:
        arg_max = np.argmax(input_arr)
        np.put(np_out_grad, arg_max, 1)
    else:
        arg_max = np.expand_dims(np.argmax(input_arr, axis=dim), axis=dim)
        np.put_along_axis(np_out_grad, arg_max, 1, axis=dim)

    return np_out, np_out_grad, input_arr


def _test_max(
    test_case, placement, sbp, np_out, np_out_grad, input_arr, shape, dim, keepdims
):
    # of result
    global_x = flow.tensor(
        input_arr,
        dtype=flow.float32,
        requires_grad=True,
        placement=flow.placement.all("cpu"),
        sbp=flow.sbp.broadcast,
    )
    if dim is None:
        of_out = flow.max(global_x)
    else:
        of_out = flow.max(global_x, dim, keepdims)[0]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()

    test_case.assertTrue(
        np.allclose(global_x.grad.numpy(), np_out_grad, 0.0001, 0.0001)
    )


class TestMaxModule(flow.unittest.TestCase):
    # backward formula is different from one of torch.
    @globaltest
    def test_eager_global_max(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_max]
        arg_dict["shape"] = [(8,), (8, 8), (8, 8, 8, 8)]
        arg_dict["dim"] = [None, 0, -1]
        arg_dict["keepdims"] = [False, True]
        for arg in GenArgList(arg_dict):
            np_out, np_out_grad, input_arr = _np_max(*arg[1:])
            np_out = (
                flow.tensor(np_out)
                .to_global(placement=flow.placement.all("cpu"), sbp=flow.sbp.broadcast,)
                .numpy()
            )
            np_out_grad = (
                flow.tensor(np_out_grad)
                .to_global(placement=flow.placement.all("cpu"), sbp=flow.sbp.broadcast,)
                .numpy()
            )
            input_arr = (
                flow.tensor(input_arr)
                .to_global(placement=flow.placement.all("cpu"), sbp=flow.sbp.broadcast,)
                .numpy()
            )
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=len(*arg[1:2])):
                    arg[0](
                        test_case,
                        placement,
                        sbp,
                        np_out,
                        np_out_grad,
                        input_arr,
                        *arg[1:]
                    )


if __name__ == "__main__":
    unittest.main()
