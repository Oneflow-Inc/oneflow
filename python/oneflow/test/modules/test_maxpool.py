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
from pkg_resources import packaging

import numpy as np
import torch as pytorch

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t


def _test_maxpool2d_channel_last(
    test_case, device, shape, kernel_size, stride, padding, dilation, ceil_mode
):
    os.environ["ONEFLOW_ENABLE_NHWC"] = "1"
    arr = np.random.randn(*shape)
    x1 = flow.tensor(arr, dtype=flow.float64, device=device)
    m1 = flow.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    y1 = m1(x1)

    x2 = pytorch.tensor(arr.transpose(0, 3, 1, 2), dtype=pytorch.float64, device=device)
    m2 = pytorch.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    y2 = m2(x2).permute(0, 2, 3, 1)
    os.environ["ONEFLOW_ENABLE_NHWC"] = "0"
    # The test fails with pytorch 1.10 but success with pytorch1.13. It should be took back after updating to pytorch1.13.
    # test_case.assertTrue(
    #     np.allclose(y1.detach().cpu().numpy(), y2.detach().cpu().numpy(), 1e-4, 1e-4)
    # )


@flow.unittest.skip_unless_1n1d()
class TestMaxPooling(flow.unittest.TestCase):
    @autotest(n=5, auto_backward=True, check_graph=True)
    def test_maxpool1d_with_random_data(test_case):
        return_indices = random().to(bool).value()
        m = torch.nn.MaxPool1d(
            kernel_size=random(4, 6).to(_size_1_t),
            stride=random(1, 3).to(_size_1_t) | nothing(),
            padding=random(1, 3).to(_size_1_t) | nothing(),
            dilation=random(2, 4).to(_size_1_t) | nothing(),
            ceil_mode=random(),
            return_indices=return_indices,
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=3, dim2=random(20, 22)).to(device)
        y = m(x)

        # NOTE(lixiang): When return_indices=False, maxpool1d will return the max indices along with the outputs,
        #   y[1] tensor has no grad_fn and cannot be backward, so only y[0] is verified here.
        if return_indices:
            return y[0]
        else:
            return y

    @autotest(n=10, auto_backward=True, check_graph=True)
    def test_maxpool2d_with_random_data(test_case):
        return_indices = random().to(bool).value()
        m = torch.nn.MaxPool2d(
            kernel_size=random(4, 6).to(_size_2_t),
            stride=random(1, 3).to(_size_2_t) | nothing(),
            padding=random(1, 3).to(_size_2_t) | nothing(),
            dilation=random(2, 4).to(_size_2_t) | nothing(),
            ceil_mode=random(),
            return_indices=return_indices,
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=4, dim2=random(20, 22), dim3=random(20, 22)).to(device)
        y = m(x)

        # NOTE(lixiang): When return_indices=False, maxpool2d will return the max indices along with the outputs,
        #   y[1] tensor has no grad_fn and cannot be backward, so only y[0] is verified here.
        if return_indices:
            return y[0]
        else:
            return y

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @autotest(n=5, auto_backward=False)
    def test_maxpool2d_with_half_data(test_case):
        return_indices = random().to(bool).value()
        m = torch.nn.MaxPool2d(
            kernel_size=random(4, 6).to(_size_2_t),
            stride=random(1, 3).to(_size_2_t) | nothing(),
            padding=random(1, 3).to(_size_2_t) | nothing(),
            dilation=random(2, 4).to(_size_2_t) | nothing(),
            ceil_mode=random(),
            return_indices=return_indices,
        )
        m.train(random())
        device = gpu_device()
        m.to(device)
        x = (
            random_tensor(ndim=4, dim2=random(20, 22), dim3=random(20, 22))
            .to(device)
            .to(torch.float16)
        )
        y = m(x)
        if return_indices:
            return y[0]
        else:
            return y

    @autotest(n=5, auto_backward=True, check_graph=True)
    def test_maxpool3d_with_random_data(test_case):
        return_indices = random().to(bool).value()
        m = torch.nn.MaxPool3d(
            kernel_size=random(4, 6).to(_size_3_t),
            stride=random(1, 3).to(_size_3_t) | nothing(),
            padding=random(1, 3).to(_size_3_t) | nothing(),
            dilation=random(2, 4).to(_size_3_t) | nothing(),
            ceil_mode=random(),
            return_indices=return_indices,
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(
            ndim=5, dim2=random(20, 22), dim3=random(20, 22), dim4=random(20, 22)
        ).to(device)
        y = m(x)

        # NOTE(lixiang): When return_indices=False, maxpool3d will return the max indices along with the outputs,
        #   y[1] tensor has no grad_fn and cannot be backward, so only y[0] is verified here.
        if return_indices:
            return y[0]
        else:
            return y

    @unittest.skipIf(
        packaging.version.parse(pytorch.__version__)
        == packaging.version.parse("1.10.0"),
        "skip when pytorch version == 1.10.0",
    )
    # NOTE:pytorch maxpool2d nhwc has bug in version of 1.10.0, so skip it in CI.
    # detail:https://github.com/pytorch/pytorch/pull/76597
    def test_maxpool2d_channel_last(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_maxpool2d_channel_last]
        arg_dict["device"] = ["cuda"]
        # CPU pool is very slow, so don't run it with CUDA
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            arg_dict["device"] = ["cpu"]
        arg_dict["shape"] = [(3, 14, 27, 3), (5, 9, 14, 10), (2, 224, 224, 3)]
        arg_dict["kernel_size"] = [3, (2, 3), (3, 4)]
        arg_dict["stride"] = [1, (1, 2), 2]
        arg_dict["padding"] = [0, (0, 1)]
        arg_dict["dilation"] = [1, (1, 2), 2]
        arg_dict["ceil_mode"] = [True, False]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


@flow.unittest.skip_unless_1n1d()
class TestMaxPoolingFunctional(flow.unittest.TestCase):
    @autotest(n=5, auto_backward=True, check_graph=True)
    def test_maxpool1d_with_random_data(test_case):
        return_indices = random().to(bool).value()
        device = random_device()
        x = random_tensor(ndim=3, dim2=random(20, 22)).to(device)
        y = torch.nn.functional.max_pool1d(
            x,
            kernel_size=random(4, 6).to(int),
            stride=random(1, 3).to(int) | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(2, 4).to(int) | nothing(),
            ceil_mode=random().to(bool),
            return_indices=return_indices,
        )

        # NOTE(lixiang): When return_indices=False, maxpool1d will return the max indices along with the outputs,
        #   y[1] tensor has no grad_fn and cannot be backward, so only y[0] is verified here.
        if return_indices:
            return y[0]
        else:
            return y

    @autotest(n=5, auto_backward=True, check_graph=True)
    def test_maxpool2d_with_random_data(test_case):
        return_indices = random().to(bool).value()
        device = random_device()
        x = random_tensor(ndim=4, dim2=random(20, 22), dim3=random(20, 22)).to(device)
        y = torch.nn.functional.max_pool2d(
            x,
            kernel_size=random(4, 6).to(int),
            stride=random(1, 3).to(int) | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(2, 4).to(int) | nothing(),
            ceil_mode=random().to(bool),
            return_indices=return_indices,
        )

        # NOTE(lixiang): When return_indices=False, maxpool2d will return the max indices along with the outputs,
        #   y[1] tensor has no grad_fn and cannot be backward, so only y[0] is verified here.
        if return_indices:
            return y[0]
        else:
            return y

    @autotest(auto_backward=True, check_graph=True)
    def test_maxpool3d_with_random_data(test_case):
        return_indices = random().to(bool).value()
        device = random_device()
        x = random_tensor(
            ndim=5, dim2=random(20, 22), dim3=random(20, 22), dim4=random(20, 22)
        ).to(device)
        y = torch.nn.functional.max_pool3d(
            x,
            kernel_size=random(4, 6).to(int),
            stride=random(1, 3).to(int) | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(2, 4).to(int) | nothing(),
            ceil_mode=random().to(bool),
            return_indices=return_indices,
        )

        # NOTE(lixiang): When return_indices=False, maxpool3d will return the max indices along with the outputs,
        #   y[1] tensor has no grad_fn and cannot be backward, so only y[0] is verified here.
        if return_indices:
            return y[0]
        else:
            return y

    @profile(torch.nn.functional.max_pool2d)
    def profile_maxpool2d(test_case):
        torch.nn.functional.max_pool2d(
            torch.ones(1, 128, 28, 28), kernel_size=3, padding=1
        )
        torch.nn.functional.max_pool2d(
            torch.ones(1, 128, 28, 28), kernel_size=3, stride=2, padding=1
        )
        torch.nn.functional.max_pool2d(
            torch.ones(16, 128, 28, 28), kernel_size=3, padding=1
        )
        torch.nn.functional.max_pool2d(
            torch.ones(16, 128, 28, 28), kernel_size=3, stride=2, padding=1
        )
        torch.nn.functional.max_pool2d(
            torch.ones(16, 128, 28, 28),
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=True,
        )
        # torch.nn.functional.max_pool2d(torch.ones(16, 128, 28, 28), kernel_size=3, dilation=2, padding=2)


if __name__ == "__main__":
    unittest.main()
