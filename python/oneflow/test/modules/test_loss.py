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

import numpy as np
import oneflow as flow
import oneflow.unittest
import torch as torch_original
from packaging import version


def generate_necessity_for_cross_entropy_or_nll_loss(dim: int, prob: bool = False):
    if dim > 5 or dim < 2:
        raise ValueError("dim should be less than 5 or greater than 1. ")
    device = random_device()
    num_classes = random(low=2).to(int)
    batch_size = random(low=10, high=100).to(int)
    ignore_index = (
        random(0, num_classes).to(int) | nothing()
        if num_classes.value() > 2 and not prob
        else nothing()
    )
    extra_dim = [random().to(int) for _ in range(dim - 2)]

    if prob:
        target_tensor = random_tensor(
            dim, batch_size, num_classes, *extra_dim, requires_grad=False,
        ).to(device)
    else:
        target_tensor = random_tensor(
            dim - 1,
            batch_size,
            *extra_dim,
            low=0,
            high=num_classes,
            dtype=int,
            requires_grad=False,
        ).to(device)
    return (
        random_tensor(dim, batch_size, num_classes, *extra_dim).to(device),
        target_tensor,
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
        random_tensor(dim, batch_size, num_classes, low=0, high=1, *extra_dim).to(
            device
        ),
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


def _test_cross_entropy_loss(dim: int, prob: bool = False):
    (
        x,
        target,
        weight,
        ignore_index,
        device,
    ) = generate_necessity_for_cross_entropy_or_nll_loss(dim, prob)
    m = torch.nn.CrossEntropyLoss(
        reduction=oneof("none", "sum", "mean", nothing()),
        ignore_index=ignore_index,
        weight=oneof(weight, nothing()),
        # TODO(wangyi): PyTorch under 1.12 has bug here, which returns wrong result when ignore_index >= 0 and label_smoothing > 0
        label_smoothing=random(low=0, high=1)
        if version.parse(torch_original.__version__) >= version.parse("1.12.0")
        else 0,
    )
    m.train(random())
    m.to(device)

    y = m(x, target)
    return y


def _test_nn_functional_cross_entropy_loss(dim: int, prob: bool):
    (
        x,
        target,
        weight,
        ignore_index,
        device,
    ) = generate_necessity_for_cross_entropy_or_nll_loss(dim, prob)
    y1 = torch.nn.functional.cross_entropy(x, target)
    y2 = torch.nn.functional.cross_entropy(x, target, weight)
    return y1 + y2


@flow.unittest.skip_unless_1n1d()
class TestCrossEntropyLossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_cross_entropy_loss_with_random_data_dim_2(test_case):
        return _test_cross_entropy_loss(2, prob=False)

    @autotest(n=5)
    def test_cross_entropy_loss_with_random_data_dim_3(test_case):
        return _test_cross_entropy_loss(3, prob=False)

    @autotest(n=5)
    def test_cross_entropy_loss_with_random_data_dim_4(test_case):
        return _test_cross_entropy_loss(4, prob=False)

    @autotest(n=5)
    def test_cross_entropy_loss_with_random_data_dim_5(test_case):
        return _test_cross_entropy_loss(5, prob=False)

    @autotest(n=5)
    def test_nn_functional_cross_entropy_with_random_data_dim(test_case):
        dim = random(2, 6).to(int).value()
        return _test_nn_functional_cross_entropy_loss(dim, prob=False)

    @autotest(n=5)
    def test_cross_entropy_prob_loss_with_random_data_dim_2(test_case):
        return _test_cross_entropy_loss(2, prob=True)

    @autotest(n=5, rtol=1e-3)
    def test_cross_entropy_prob_loss_with_random_data_dim_3(test_case):
        return _test_cross_entropy_loss(3, prob=True)

    @autotest(n=5)
    def test_cross_entropy_prob_loss_with_random_data_dim_4(test_case):
        return _test_cross_entropy_loss(4, prob=True)

    @autotest(n=5)
    def test_cross_entropy_prob_loss_with_random_data_dim_5(test_case):
        return _test_cross_entropy_loss(5, prob=True)

    @autotest(n=5)
    def test_nn_functional_prob_cross_entropy_with_random_data_dim(test_case):
        dim = random(2, 6).to(int).value()
        return _test_nn_functional_cross_entropy_loss(dim, prob=True)


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
    pos_weight_for_testing_broadcast = random_tensor(
        1, 1, low=1, high=3, requires_grad=False,
    ).to(device)
    if with_logits:
        m = torch.nn.BCEWithLogitsLoss(
            weight=oneof(weight, nothing()),
            pos_weight=oneof(pos_weight, pos_weight_for_testing_broadcast, nothing()),
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


def _test_nn_functional_binary_cross_entropy_with_logits_different_dtype_float_first(
    test_case, shape, reduction, device
):
    def compare(a, b):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(),
                b.detach().cpu().numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
        )

    arr = np.random.randn(*shape)

    flow_pred_mask = flow.Tensor(arr).float().to(device)
    flow_pred_mask.requires_grad = True
    flow_gt_mask = flow.Tensor(arr).double().to(device)
    flow_loss = flow.nn.functional.binary_cross_entropy_with_logits(
        flow_pred_mask, flow_gt_mask, reduction=reduction
    )
    flow_loss.sum().backward()
    torch_pred_mask = torch_original.Tensor(arr).float().to(device)
    torch_pred_mask.requires_grad = True
    torch_gt_mask = torch_original.Tensor(arr).double().to(device)
    torch_loss = torch_original.nn.functional.binary_cross_entropy_with_logits(
        torch_pred_mask, torch_gt_mask, reduction=reduction
    )
    torch_loss.sum().backward()
    compare(flow_loss, torch_loss)
    compare(flow_pred_mask.grad.data, torch_pred_mask.grad.data)


def _test_nn_functional_binary_cross_entropy_with_logits_different_dtype_double_first(
    test_case, shape, reduction, device
):
    def compare(a, b):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(),
                b.detach().cpu().numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
        )

    arr = np.random.randn(*shape)

    flow_pred_mask = flow.Tensor(arr).double().to(device)
    flow_pred_mask.requires_grad = True
    flow_gt_mask = flow.Tensor(arr).float().to(device)
    flow_loss = flow.nn.functional.binary_cross_entropy_with_logits(
        flow_pred_mask, flow_gt_mask, reduction=reduction
    )
    flow_loss.sum().backward()
    torch_pred_mask = torch_original.Tensor(arr).double().to(device)
    torch_pred_mask.requires_grad = True
    torch_gt_mask = torch_original.Tensor(arr).float().to(device)
    torch_loss = torch_original.nn.functional.binary_cross_entropy_with_logits(
        torch_pred_mask, torch_gt_mask, reduction=reduction
    )
    torch_loss.sum().backward()
    compare(flow_loss, torch_loss)
    compare(flow_pred_mask.grad.data, torch_pred_mask.grad.data)


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

    @autotest(n=5)
    def test_nn_functional_binary_cross_entropy_with_logits_different_dtype(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_nn_functional_binary_cross_entropy_with_logits_different_dtype_float_first,
            _test_nn_functional_binary_cross_entropy_with_logits_different_dtype_double_first,
        ]
        arg_dict["shape"] = [(24, 16, 80), (42, 160), (4, 54, 32, 56)]
        arg_dict["reduction"] = ["sum", "mean", "none"]
        arg_dict["device"] = ["cpu", "cuda"]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


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

    @autotest(n=5)
    def _test_nn_functional_l1_loss(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape

        x = random_tensor(len(shape), *shape).to(device)
        target = random_tensor(len(shape), *shape, requires_grad=False).to(device)

        y = torch.nn.functional.l1_loss(
            x, target, reduction=oneof("none", "sum", "mean", nothing())
        )
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

    @autotest(n=5)
    def _test_nn_functional_mse_loss(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape

        x = random_tensor(len(shape), *shape).to(device)
        target = random_tensor(len(shape), *shape, requires_grad=False).to(device)

        y = torch.nn.functional.mse_loss(
            x, target, reduction=oneof("none", "sum", "mean", nothing())
        )
        return y


@flow.unittest.skip_unless_1n1d()
class TestKLDivLossModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_kldiv_loss_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape

        x = random_tensor(len(shape), low=0, *shape).to(device)
        target = random_tensor(len(shape), low=0, *shape, requires_grad=False).to(
            device
        )

        m = torch.nn.KLDivLoss(
            reduction=oneof("none", "sum", "mean", "batchmean", nothing()),
            log_target=oneof(True, False, nothing()),
        )
        m.train(random())
        m.to(device)

        y = m(x, target)
        return y

    @autotest(n=5)
    def test_nn_functional_kl_div(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x = random_tensor(len(shape), low=0, *shape).to(device)
        target = random_tensor(len(shape), low=0, *shape, requires_grad=False).to(
            device
        )
        y = torch.nn.functional.kl_div(
            x,
            target,
            reduction=oneof("none", "sum", "mean", "batchmean", nothing()),
            log_target=oneof(True, False, nothing()),
        )
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
