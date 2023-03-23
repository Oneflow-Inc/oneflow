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


def _test_batchnorm2d_infer(
    test_case, shape, track_running_stats, affine, device, dtype
):
    arr = np.random.randn(*shape) * 3
    x1 = flow.tensor(arr, device=flow.device(device), dtype=dtype)
    x2 = flow.tensor(arr, device="cpu", dtype=dtype)
    m1 = (
        flow.nn.BatchNorm2d(
            num_features=int(x1.shape[1]),
            track_running_stats=track_running_stats,
            affine=affine,
            eps=1e-05,
            momentum=0.1,
        )
        .eval()
        .to(flow.device(device))
    )

    m2 = (
        flow.nn.BatchNorm2d(
            num_features=int(x2.shape[1]),
            track_running_stats=track_running_stats,
            affine=affine,
            eps=1e-05,
            momentum=0.1,
        )
        .eval()
        .to("cpu")
    )

    mlu_out = m1(x1)
    cpu_out = m2(x2)

    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out.numpy(), 0.001, 0.001))


def _test_batchnorm2d_train(
    test_case, shape, track_running_stats, affine, device, dtype
):
    arr = np.random.randn(*shape)
    x1 = flow.tensor(arr, device=flow.device(device), dtype=dtype, requires_grad=True)
    x2 = flow.tensor(arr, device="cpu", dtype=dtype, requires_grad=True)
    m1 = (
        flow.nn.BatchNorm2d(
            num_features=int(x1.shape[1]),
            track_running_stats=track_running_stats,
            affine=affine,
        )
        .train()
        .to(flow.device(device))
    )

    m2 = (
        flow.nn.BatchNorm2d(
            num_features=int(x2.shape[1]),
            track_running_stats=track_running_stats,
            affine=affine,
        )
        .train()
        .to("cpu")
    )

    mlu_out = m1(x1)
    cpu_out = m2(x2)
    # forward
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out.numpy(), 0.001, 0.001))


@flow.unittest.skip_unless_1n1d()
class TestBatchNormCambriconModule(flow.unittest.TestCase):
    def test_batchnorm2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_batchnorm2d_infer,
            _test_batchnorm2d_train,
        ]
        arg_dict["shape"] = [(2, 3, 4, 5), (1, 2, 3, 4), (5, 6, 7, 8)]
        arg_dict["track_running_stats"] = [True, False]
        arg_dict["affine"] = [True, False]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [flow.float32]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
