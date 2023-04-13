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

from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@autotest(n=1, check_graph=True)
def _test_global_where(test_case, placement, sbp):
    x = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    y = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    condition = random_tensor(ndim=2, dim0=8, dim1=16, high=2, dtype=int).to_global(
        placement, sbp
    )

    condition = condition.to(torch.bool)

    z = torch.where(condition, x, y)
    return z


@autotest(n=1, check_graph=True)
def _test_global_where_broadcast(test_case, placement, sbp):
    x = random_tensor(ndim=3, dim0=8, dim1=16, dim2=1).to_global(placement, sbp)
    y = random_tensor(ndim=3, dim0=8, dim1=16, dim2=8).to_global(placement, sbp)
    condition = random_tensor(
        ndim=3, dim0=8, dim1=16, dim2=1, high=2, dtype=int
    ).to_global(placement, sbp)

    condition = condition.to(torch.bool)

    z = torch.where(condition, x, y)
    return z


@autotest(n=1, check_graph=True)
def _test_global_where_scalar(test_case, placement, sbp):
    x = random_tensor(ndim=0).to_global(placement, sbp)
    y = random_tensor(ndim=0).to_global(placement, sbp)
    condition = random_tensor(ndim=0, high=2, dtype=int).to_global(placement, sbp)

    condition = condition.to(torch.bool)

    z = torch.where(condition, x, y)
    return z


# Close auto_backward because pytorch raise error:
# PyTorch error: element 0 of tensors does not require grad and does not have a grad_fn
# Not check graph because of one reason:
# Reason 1, lazy tensor cannot call .numpy(), tensor.numpy() is not allowed to called in nn.Graph.build(*args) or called by lazy tensor.
# Please refer to File "python/oneflow/nn/modules/nonzero.py", line 29, in nonzero_op. Because nonzero_op is called by where.
@autotest(n=1, auto_backward=False, check_graph="ValidatedFalse")
def _test_where_x_y_none(test_case, placement, sbp):
    condition = random_tensor(ndim=2, dim0=8, dim1=8, low=-1, high=1).to_global(
        placement, sbp
    )
    y = torch.where(condition)
    return y[0], y[1]


@autotest(n=1, check_graph=True)
def _test_global_where_tensor_with_0dim_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = random_tensor(ndim=0).to_global(placement, sbp)
    y = random_tensor(ndim=0).to_global(placement, sbp)
    return torch.where(cond > 0, x, y)


@autotest(n=1, check_graph=True)
def _test_flow_where_tensor_broadcast_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=3, dim0=8, dim1=16, dim2=8).to_global(placement, sbp)
    x = random_tensor(ndim=3, dim0=8, dim1=1, dim2=8).to_global(placement, sbp)
    y = random_tensor(ndim=3, dim0=8, dim1=16, dim2=1).to_global(placement, sbp)
    return torch.where(cond > 0, x, y)


@autotest(n=1, check_graph=True)
def _test_flow_where_scalar_x_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = random().to(float)
    y = (
        random_tensor(ndim=2, dim0=8, dim1=16, dtype=float)
        .to_global(placement, sbp)
        .to(torch.float64)
    )
    return torch.where(cond > 0, x, y)


@autotest(n=1, check_graph=True)
def _test_flow_where_scalar_x_broadcast_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=1, dim1=16).to_global(placement, sbp)
    x = random().to(float)
    y = (
        random_tensor(ndim=2, dim0=8, dim1=1, dtype=float)
        .to_global(placement, sbp)
        .to(torch.float64)
    )
    return torch.where(cond > 0, x, y)


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_flow_where_scalar_x_int_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = random().to(int)
    y = random_tensor(ndim=2, dim0=8, dim1=16, dtype=int).to_global(placement, sbp)
    return torch.where(cond > 0, x, y)


@autotest(n=1, check_graph=True)
def _test_flow_where_scalar_y_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = (
        random_tensor(ndim=2, dim0=8, dim1=16, dtype=float)
        .to_global(placement, sbp)
        .to(torch.float64)
    )
    y = random().to(float)
    return torch.where(cond > 0, x, y)


@autotest(n=1, check_graph=True)
def _test_flow_where_scalar_y_broadcast_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=1, dim1=16).to_global(placement, sbp)
    x = (
        random_tensor(ndim=2, dim0=8, dim1=1, dtype=float)
        .to_global(placement, sbp)
        .to(torch.float64)
    )
    y = random().to(float)
    return torch.where(cond > 0, x, y)


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_flow_where_scalar_y_int_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = random_tensor(ndim=2, dim0=8, dim1=16, dtype=int).to_global(placement, sbp)
    y = random().to(int)
    return torch.where(cond > 0, x, y)


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_flow_where_tensor_bool_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp).to(torch.bool)
    y = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp).to(torch.bool)
    return torch.where(cond > 0, x, y)


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_flow_where_tensor_broadcast_bool_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = random_tensor(ndim=2, dim0=1, dim1=16).to_global(placement, sbp).to(torch.bool)
    y = random_tensor(ndim=2, dim0=8, dim1=1).to_global(placement, sbp).to(torch.bool)
    return torch.where(cond > 0, x, y)


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_flow_where_scalar_x_bool_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = random().to(bool)
    y = (
        random_tensor(ndim=2, dim0=8, dim1=16, dtype=float)
        .to_global(placement, sbp)
        .to(torch.bool)
    )
    return torch.where(cond > 0, x, y)


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_flow_where_scalar_x_broadcast_bool_with_random_data(
    test_case, placement, sbp
):
    cond = random_tensor(ndim=2, dim0=1, dim1=16).to_global(placement, sbp)
    x = random().to(bool)
    y = (
        random_tensor(ndim=2, dim0=8, dim1=1, dtype=float)
        .to_global(placement, sbp)
        .to(torch.bool)
    )
    return torch.where(cond > 0, x, y)


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_flow_where_scalar_y_bool_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = (
        random_tensor(ndim=2, dim0=8, dim1=16, dtype=float)
        .to_global(placement, sbp)
        .to(torch.bool)
    )
    y = random().to(bool)
    return torch.where(cond > 0, x, y)


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_flow_where_scalar_y_broadcast_bool_with_random_data(
    test_case, placement, sbp
):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = (
        random_tensor(ndim=2, dim0=8, dim1=1, dtype=float)
        .to_global(placement, sbp)
        .to(torch.bool)
    )
    y = random().to(bool)
    return torch.where(cond > 0, x, y)


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_flow_where_scalar_xy_bool_with_random_data(test_case, placement, sbp):
    cond = random_tensor(ndim=2, dim0=8, dim1=16).to_global(placement, sbp)
    x = random().to(bool)
    y = random().to(bool)
    return torch.where(cond > 0, x, y)


class TestGlobalWhere(flow.unittest.TestCase):
    @globaltest
    def test_global_where(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_where(test_case, placement, sbp)

    @globaltest
    def test_global_where_broadcast(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_where_broadcast(test_case, placement, sbp)

    @globaltest
    def test_global_where_scalar(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_global_where_scalar(test_case, placement, sbp)

    @globaltest
    def test_where_x_y_none(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_where_x_y_none(test_case, placement, sbp)

    @globaltest
    def test_global_where_tensor_with_0dim_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_global_where_tensor_with_0dim_data(test_case, placement, sbp)

    @globaltest
    def test_flow_where_tensor_broadcast_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_flow_where_tensor_broadcast_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_scalar_x_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_x_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_flow_where_scalar_x_broadcast_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_x_broadcast_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_scalar_x_int_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_x_int_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_scalar_y_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_y_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_flow_where_scalar_y_broadcast_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_y_broadcast_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_scalar_y_int_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_y_int_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_tensor_bool_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_flow_where_tensor_bool_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_flow_where_tensor_broadcast_bool_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_tensor_broadcast_bool_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_scalar_x_bool_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_x_bool_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_scalar_x_broadcast_bool_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_x_broadcast_bool_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_scalar_y_bool_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_y_bool_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_scalar_y_broadcast_bool_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_y_broadcast_bool_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_where_scalar_xy_bool_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, except_split=True):
                _test_flow_where_scalar_xy_bool_with_random_data(
                    test_case, placement, sbp
                )


if __name__ == "__main__":
    unittest.main()
