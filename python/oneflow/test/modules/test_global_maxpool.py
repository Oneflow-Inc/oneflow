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
from pkg_resources import packaging
import oneflow as flow
import torch as ori_torch
import oneflow.unittest
from collections import OrderedDict
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t


@autotest(n=1, check_graph=True)
def _test_maxpool1d_functional(test_case, placement, sbp):
    return_indices = random().to(bool).value()
    dim0 = random(1, 4).to(int).value() * 8
    dim1 = random(1, 4).to(int).value() * 8
    x = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(20, 22)).to_global(
        placement, sbp
    )
    y = torch.nn.functional.max_pool1d(
        x,
        kernel_size=random(4, 6).to(int),
        stride=random(1, 3).to(int),
        padding=random(1, 3).to(int),
        dilation=random(2, 4).to(int),
        ceil_mode=random().to(bool),
        return_indices=return_indices,
    )
    if return_indices:
        return y[0]
    else:
        return y


@autotest(n=1, check_graph=True)
def _test_maxpool2d_functional(test_case, placement, sbp):
    return_indices = random().to(bool).value()
    dim0 = random(1, 4).to(int).value() * 8
    dim1 = random(1, 4).to(int).value() * 8
    x = random_tensor(
        ndim=4, dim0=dim0, dim1=dim1, dim2=random(20, 22), dim3=random(20, 22)
    ).to_global(placement, sbp)
    y = torch.nn.functional.max_pool2d(
        x,
        kernel_size=random(4, 6).to(int),
        stride=random(1, 3).to(int),
        padding=random(1, 3).to(int),
        dilation=random(2, 4).to(int),
        ceil_mode=random().to(bool),
        return_indices=return_indices,
    )

    if return_indices:
        return y[0]
    else:
        return y


@autotest(n=1, check_graph=True)
def _test_maxpool3d_functional(test_case, placement, sbp):
    return_indices = random().to(bool).value()
    dim0 = random(high=4).to(int).value() * 8
    dim1 = random(high=4).to(int).value() * 8
    x = random_tensor(
        ndim=5,
        dim0=dim0,
        dim1=dim1,
        dim2=random(10, 12),
        dim3=random(10, 12),
        dim4=random(10, 12),
    ).to_global(placement, sbp)
    y = torch.nn.functional.max_pool3d(
        x,
        kernel_size=random(4, 6).to(int),
        stride=random(1, 3).to(int),
        padding=random(1, 3).to(int),
        dilation=random(2, 4).to(int),
        ceil_mode=random().to(bool),
        return_indices=return_indices,
    )

    if return_indices:
        return y[0]
    else:
        return y


@autotest(n=1, check_graph=True)
def _test_maxpool1d(test_case, placement, sbp):
    return_indices = random().to(bool).value()
    dim0 = random(1, 4).to(int).value() * 8
    dim1 = random(1, 4).to(int).value() * 8
    m = torch.nn.MaxPool1d(
        kernel_size=random(4, 6).to(_size_1_t),
        stride=random(1, 3).to(_size_1_t),
        padding=random(1, 3).to(_size_1_t),
        dilation=random(2, 4).to(_size_1_t),
        ceil_mode=random(),
        return_indices=return_indices,
    )
    m.train(random())
    x = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(20, 22)).to_global(
        placement, sbp
    )
    y = m(x)
    if return_indices:
        return y[0]
    else:
        return y


@autotest(n=1, check_graph=True)
def _test_maxpool2d(test_case, placement, sbp):
    return_indices = random().to(bool).value()
    dim0 = random(1, 3).to(int).value() * 8
    dim1 = random(1, 3).to(int).value() * 8
    m = torch.nn.MaxPool2d(
        kernel_size=random(4, 6).to(_size_2_t),
        stride=random(1, 3).to(_size_2_t),
        padding=random(1, 3).to(_size_2_t),
        dilation=random(2, 4).to(_size_2_t),
        ceil_mode=random(),
        return_indices=return_indices,
    )
    m.train(random())
    x = random_tensor(
        ndim=4, dim0=dim0, dim1=dim1, dim2=random(20, 22), dim3=random(20, 22)
    ).to_global(placement, sbp)
    y = m(x)
    if return_indices:
        return y[0]
    else:
        return y


@autotest(n=1, check_graph=True)
def _test_maxpool3d(test_case, placement, sbp):
    return_indices = random().to(bool).value()
    dim0 = random(high=4).to(int).value() * 8
    dim1 = random(high=4).to(int).value() * 8
    m = torch.nn.MaxPool3d(
        kernel_size=random(4, 6).to(_size_3_t),
        stride=random(1, 3).to(_size_3_t),
        padding=random(1, 3).to(_size_3_t),
        dilation=random(2, 4).to(_size_3_t),
        ceil_mode=random(),
        return_indices=return_indices,
    )
    m.train(random())
    x = random_tensor(
        ndim=5,
        dim0=dim0,
        dim1=dim1,
        dim2=random(10, 12),
        dim3=random(10, 12),
        dim4=random(10, 12),
    ).to_global(placement, sbp)
    y = m(x)

    if return_indices:
        return y[0]
    else:
        return y


def _test_maxpool2d_channel_last(
    test_case, placement, sbp, shape, kernel_size, stride, padding, dilation, ceil_mode
):
    os.environ["ONEFLOW_ENABLE_NHWC"] = "1"

    tensor = random_tensor(len(shape), *shape, requires_grad=False).to_global(
        placement, sbp
    )
    # oneflow result
    x1 = tensor.oneflow
    m1 = flow.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    y1 = m1(x1)

    # pytorch result
    x2 = tensor.pytorch.permute(0, 3, 1, 2).to(placement.type)
    m2 = ori_torch.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    y2 = m2(x2).permute(0, 2, 3, 1)
    os.environ["ONEFLOW_ENABLE_NHWC"] = "1"

    # It should be added after updating to torch1.13
    # test_case.assertTrue(
    #     np.allclose(y1.detach().cpu().numpy(), y2.detach().cpu().numpy(), 1e-4, 1e-4)
    # )


class TestMaxPool(flow.unittest.TestCase):
    @globaltest
    def test_maxpool(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_maxpool1d_functional(test_case, placement, sbp)
                _test_maxpool2d_functional(test_case, placement, sbp)
                _test_maxpool3d_functional(test_case, placement, sbp)
                _test_maxpool1d(test_case, placement, sbp)
                _test_maxpool2d(test_case, placement, sbp)
                _test_maxpool3d(test_case, placement, sbp)

    @globaltest
    @unittest.skipIf(
        packaging.version.parse(ori_torch.__version__)
        == packaging.version.parse("1.10.0"),
        "skip when pytorch version == 1.10.0",
    )
    # NOTE:pytorch maxpool2d nhwc has bug in version of 1.10.0, so skip it in CI.
    # detail:https://github.com/pytorch/pytorch/pull/76597
    def test_maxpool2d_channel_last(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_maxpool2d_channel_last]
        arg_dict["shape"] = [(1, 16, 16, 3), (2, 224, 224, 3)]
        arg_dict["kernel_size"] = [3, (2, 3)]
        arg_dict["stride"] = [1, (1, 2)]
        arg_dict["padding"] = [0, (0, 1)]
        arg_dict["dilation"] = [1, 2]
        arg_dict["ceil_mode"] = [True, False]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, valid_split_axis=[1, 2]):
                    arg[0](test_case, placement, sbp, *arg[1:])


if __name__ == "__main__":
    unittest.main()
