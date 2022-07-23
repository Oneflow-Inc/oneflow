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
from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


def generate_necessity_for_cross_entropy_or_nll_loss(dim: int):
    if dim > 5 or dim < 2:
        raise ValueError("dim should be less than 5 or greater than 1. ")
    device = random_device()
    num_classes = random(low=2).to(int)
    batch_size = random(low=10, high=100).to(int)
    ignore_index = (
        random(0, num_classes).to(int) | nothing()
        if num_classes.value() > 2
        else nothing()
    )
    extra_dim = [random().to(int) for _ in range(dim - 2)]
    return (
        random_tensor(dim, batch_size, num_classes, *extra_dim).to(device),
        random_tensor(
            dim - 1,
            batch_size,
            *extra_dim,
            low=0,
            high=num_classes,
            dtype=int,
            requires_grad=False,
        ).to(device),
        random_tensor(1, num_classes, low=0, high=3, requires_grad=False).to(device),
        ignore_index,
        device,
    )


def generate_necessity_for_bce_loss(dim: int):
    if dim > 5 or dim < 2:
        raise ValueError("dim should be less than 6 or greater than 1. ")
    device = random_device()
    num_classes = random(low=3).to(int)
    batch_size = random(low=10, high=100).to(int)
    extra_dim = [random().to(int) for _ in range(dim - 2)]
    return (
        random_tensor(dim, batch_size, num_classes, *extra_dim).to(device),
        random_tensor(
            dim,
            batch_size,
            num_classes,
            *extra_dim,
            low=0,
            high=num_classes,
            requires_grad=False,
        ).to(device),
        random_tensor(
            dim, batch_size, num_classes, *extra_dim, low=0, high=3, requires_grad=False
        ).to(device),
        random_tensor(
            1,
            extra_dim[-1] if dim > 2 else num_classes,
            low=1,
            high=3,
            requires_grad=False,
        ).to(device),
        device,
    )


def _test_cross_entropy_loss(dim=int):
    (
        x,
        target,
        weight,
        ignore_index,
        device,
    ) = generate_necessity_for_cross_entropy_or_nll_loss(dim)
    m = torch.nn.CrossEntropyLoss(
        reduction=oneof("none", "sum", "mean", nothing()),
        ignore_index=ignore_index,
        weight=oneof(weight, nothing()),
    )
    m.train(random())
    m.to(device)

    y = m(x, target)
    return y


def _test_nn_functional_cross_entropy_loss(dim=int):
    (
        x,
        target,
        weight,
        ignore_index,
        device,
    ) = generate_necessity_for_cross_entropy_or_nll_loss(dim)
    y1 = torch.nn.functional.cross_entropy(x, target)
    y2 = torch.nn.functional.cross_entropy(x, target, weight)
    return y1 + y2


@flow.unittest.skip_unless_1n1d()
class TestCrossEntropyLossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_cross_entropy_loss_with_random_data_dim_2(test_case):
        return _test_cross_entropy_loss(2)

    @autotest(n=5)
    def test_cross_entropy_loss_with_random_data_dim_3(test_case):
        return _test_cross_entropy_loss(3)

    @autotest(n=5)
    def test_cross_entropy_loss_with_random_data_dim_4(test_case):
        return _test_cross_entropy_loss(4)

    @autotest(n=5)
    def test_cross_entropy_loss_with_random_data_dim_5(test_case):
        return _test_cross_entropy_loss(5)

    @autotest(n=5)
    def test_nn_functional_cross_entropy_with_random_data_dim(test_case):
        dim = random(2, 6).to(int).value()
        return _test_nn_functional_cross_entropy_loss(dim)


def _test_nll_loss(dim=int):
    (
        x,
        target,
        weight,
        ignore_index,
        device,
    ) = generate_necessity_for_cross_entropy_or_nll_loss(dim)
    m = torch.nn.NLLLoss(
        weight=oneof(weight, nothing()),
        reduction=oneof("none", "sum", "mean", nothing()),
        ignore_index=ignore_index,
    )
    m.train(random())
    m.to(device)

    y = m(x, target)
    return y


@flow.unittest.skip_unless_1n1d()
class TestNLLLossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_nll_loss_with_random_data_dim_2(test_case):
        return _test_nll_loss(2)

    @autotest(n=5)
    def test_nll_loss_with_random_data_dim_3(test_case):
        return _test_nll_loss(3)

    @autotest(n=5)
    def test_nll_loss_with_random_data_dim_4(test_case):
        return _test_nll_loss(4)

    @autotest(n=5)
    def test_nll_loss_with_random_data_dim_5(test_case):
        return _test_nll_loss(5)


def _test_bce_loss(dim=int, with_logits: bool = False):
    x, target, weight, pos_weight, device = generate_necessity_for_bce_loss(dim)

    m = torch.nn.BCELoss(
        weight=oneof(weight, nothing()),
        reduction=oneof("none", "sum", "mean", nothing()),
    )
    if with_logits:
        m = torch.nn.BCEWithLogitsLoss(
            weight=oneof(weight, nothing()),
            pos_weight=oneof(pos_weight, nothing()),
            reduction=oneof("none", "sum", "mean", nothing()),
        )
    m.train(random())
    m.to(device)

    y = m(x, target)
    return y


def _test_nn_functional_binary_cross_entropy(dim=int):
    (x, target, weight, pos_weight, device) = generate_necessity_for_bce_loss(dim)
    y = torch.nn.functional.binary_cross_entropy(
        x,
        target,
        weight=oneof(weight, nothing()),
        reduction=oneof("none", "sum", "mean", nothing()),
        pos_weight=oneof(pos_weight, nothing()),
    )
    return y


def _test_nn_functional_binary_cross_entropy_with_logits(dim=int):
    (x, target, weight, pos_weight, device) = generate_necessity_for_bce_loss(dim)
    y = torch.nn.functional.binary_cross_entropy_with_logits(
        x,
        target,
        weight=oneof(weight, nothing()),
        reduction=oneof("none", "sum", "mean", nothing()),
    )
    return y


@flow.unittest.skip_unless_1n1d()
class TestBCELossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_bce_loss_with_random_data_dim_2(test_case):
        return _test_bce_loss(2)

    @autotest(n=5)
    def test_bce_loss_with_random_data_dim_3(test_case):
        return _test_bce_loss(3)

    @autotest(n=5)
    def test_bce_loss_with_random_data_dim_4(test_case):
        return _test_bce_loss(4)

    @autotest(n=5)
    def test_bce_loss_with_random_data_dim_5(test_case):
        return _test_bce_loss(5)

    @autotest(n=5)
    def test_nn_functional_binary_cross_entropy(test_case):
        dim = random(2, 6).to(int).value()
        return _test_nn_functional_binary_cross_entropy(dim)


@flow.unittest.skip_unless_1n1d()
class TestBCEWithLogitsLossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_bce_with_logits_loss_with_random_data_dim_2(test_case):
        return _test_bce_loss(2, True)

    @autotest(n=5)
    def test_bce_with_logits_loss_with_random_data_dim_3(test_case):
        return _test_bce_loss(3, True)

    @autotest(n=5)
    def test_bce_with_logits_loss_with_random_data_dim_4(test_case):
        return _test_bce_loss(4, True)

    @autotest(n=5)
    def test_bce_with_logits_loss_with_random_data_dim_5(test_case):
        return _test_bce_loss(5, True)

    @autotest(n=5)
    def test_nn_functional_binary_cross_entropy_with_logits(test_case):
        dim = random(2, 6).to(int).value()
        return _test_nn_functional_binary_cross_entropy_with_logits(dim)


@flow.unittest.skip_unless_1n1d()
class TestL1LossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_l1_loss_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape

        x = random_tensor(len(shape), *shape).to(device)
        target = random_tensor(len(shape), *shape, requires_grad=False).to(device)

        m = torch.nn.L1Loss(reduction=oneof("none", "sum", "mean", nothing()))
        m.train(random())
        m.to(device)

        y = m(x, target)
        return y


@flow.unittest.skip_unless_1n1d()
class TestSmoothL1LossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_smooth_l1_loss_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape

        x = random_tensor(len(shape), *shape).to(device)
        target = random_tensor(len(shape), *shape, requires_grad=False).to(device)

        m = torch.nn.SmoothL1Loss(
            reduction=oneof("none", "sum", "mean", nothing()), beta=oneof(0, 0.5, 1)
        )
        m.train(random())
        m.to(device)

        y = m(x, target)
        return y


@flow.unittest.skip_unless_1n1d()
class TestMSELossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_mse_loss_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape

        x = random_tensor(len(shape), *shape).to(device)
        target = random_tensor(len(shape), *shape, requires_grad=False).to(device)

        m = torch.nn.MSELoss(reduction=oneof("none", "sum", "mean", nothing()))
        m.train(random())
        m.to(device)

        y = m(x, target)
        return y


@flow.unittest.skip_unless_1n1d()
class TestKLDivLossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_kldiv_loss_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape

        x = random_tensor(len(shape), *shape).to(device)
        target = random_tensor(len(shape), *shape, requires_grad=False).to(device)

        m = torch.nn.KLDivLoss(
            reduction=oneof("none", "sum", "mean", nothing()),
            log_target=oneof(True, False),
        )
        m.train(random())
        m.to(device)

        y = m(x, target)
        return y


@flow.unittest.skip_unless_1n1d()
class TestMarginRankingLossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_margin_ranking_loss_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape

        x1 = random_tensor(len(shape), *shape).to(device)
        x2 = random_tensor(len(shape), *shape).to(device)
        target = random_tensor(len(shape), *shape, requires_grad=False).to(device)

        m = torch.nn.MarginRankingLoss(
            margin=oneof(0.0, 0.3, 10),
            reduction=oneof("none", "sum", "mean", nothing()),
        )
        m.train(random())
        m.to(device)

        y = m(x1, x2, target)
        return y


if __name__ == "__main__":
    unittest.main()
