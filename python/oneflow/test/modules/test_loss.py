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
import torch as pytorch
import numpy as np
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


def test_cross_entropy_loss(dim=int):
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


@flow.unittest.skip_unless_1n1d()
class TestCrossEntropyLossModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_cross_entropy_loss_with_random_data_dim_2(test_case):
        return test_cross_entropy_loss(2)

    @autotest(check_graph=True)
    def test_cross_entropy_loss_with_random_data_dim_3(test_case):
        return test_cross_entropy_loss(3)

    @autotest(check_graph=True)
    def test_cross_entropy_loss_with_random_data_dim_4(test_case):
        return test_cross_entropy_loss(4)

    @autotest(check_graph=True)
    def test_cross_entropy_loss_with_random_data_dim_5(test_case):
        return test_cross_entropy_loss(5)


def test_nll_loss(dim=int):
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
    @autotest(check_graph=True)
    def test_nll_loss_with_random_data_dim_2(test_case):
        return test_nll_loss(2)

    @autotest(check_graph=True)
    def test_nll_loss_with_random_data_dim_3(test_case):
        return test_nll_loss(3)

    @autotest(check_graph=True)
    def test_nll_loss_with_random_data_dim_4(test_case):
        return test_nll_loss(4)

    @autotest(check_graph=True)
    def test_nll_loss_with_random_data_dim_5(test_case):
        return test_nll_loss(5)


def test_bce_loss(dim=int, with_logits: bool = False):
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


@flow.unittest.skip_unless_1n1d()
class TestBCELossModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_bce_loss_with_random_data_dim_2(test_case):
        return test_bce_loss(2)

    @autotest(check_graph=True)
    def test_bce_loss_with_random_data_dim_3(test_case):
        return test_bce_loss(3)

    @autotest(check_graph=True)
    def test_bce_loss_with_random_data_dim_4(test_case):
        return test_bce_loss(4)

    @autotest(check_graph=True)
    def test_bce_loss_with_random_data_dim_5(test_case):
        return test_bce_loss(5)


@flow.unittest.skip_unless_1n1d()
class TestBCEWithLogitsLossModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_bce_with_logits_loss_with_random_data_dim_2(test_case):
        return test_bce_loss(2, True)

    @autotest(check_graph=True)
    def test_bce_with_logits_loss_with_random_data_dim_3(test_case):
        return test_bce_loss(3, True)

    @autotest(check_graph=True)
    def test_bce_with_logits_loss_with_random_data_dim_4(test_case):
        return test_bce_loss(4, True)

    @autotest(check_graph=True)
    def test_bce_with_logits_loss_with_random_data_dim_5(test_case):
        return test_bce_loss(5, True)


@flow.unittest.skip_unless_1n1d()
class TestL1LossModule(flow.unittest.TestCase):
    @autotest()
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
    @autotest()
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
    @autotest()
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
    @autotest()
    def test_kldiv_loss_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape

        x = random_tensor(len(shape), *shape).to(device)
        target = random_tensor(len(shape), *shape, requires_grad=oneof(True, False)).to(device)

        m = torch.nn.KLDivLoss(
            reduction=oneof("none", "sum", "mean", "batchmean", nothing()),
            log_target=oneof(True, False),
        )
        m.train(random())
        m.to(device)

        y = m(x, target)
        return y


@flow.unittest.skip_unless_1n1d()
class TestMarginRankingLossModule(flow.unittest.TestCase):
    @autotest()
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


class SoftTargetCrossEntropy(pytorch.nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: pytorch.Tensor, target: pytorch.Tensor) -> pytorch.Tensor:
        loss = pytorch.sum(
            -target * pytorch.nn.functional.log_softmax(x, dim=-1), dim=-1
        )
        return loss.mean()


def generate_necessity_for_soft_target_cross_entropy(dim: int):
    if dim > 5 or dim < 2:
        raise ValueError("dim should be less than 5 or greater than 1. ")
    device = random_device()
    num_classes = random(low=2).to(int)
    batch_size = random(low=10, high=100).to(int)
    extra_dim = [random().to(int) for _ in range(dim - 2)]

    def get_tensor_tuple(obtained_random_tensor, requires_grad=True):
        pytorch_tensor = obtained_random_tensor.value().requires_grad_(requires_grad)
        flow_tensor = flow.tensor(
            pytorch_tensor.detach().cpu().numpy(), requires_grad=requires_grad,
        )
        return pytorch_tensor, flow_tensor

    return (
        *get_tensor_tuple(
            random_tensor(dim, batch_size, num_classes, *extra_dim).to(device)
        ),
        *get_tensor_tuple(
            random_tensor(dim, batch_size, num_classes, *extra_dim).to(device), False,
        ),
    )


def test_soft_target_cross_entropy(dim: int):
    (
        pytorch_input,
        oneflow_input,
        pytorch_target,
        oneflow_target,
    ) = generate_necessity_for_soft_target_cross_entropy(4)
    pytorch_out = SoftTargetCrossEntropy()(pytorch_input, pytorch_target)
    pytorch_out.sum().backward()
    oneflow_out = flow.nn.SoftTargetCrossEntropy()(oneflow_input, oneflow_target)
    oneflow_out.sum().backward()
    assert np.allclose(
        oneflow_out.numpy(), pytorch_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    assert np.allclose(
        oneflow_input.grad.numpy(),
        pytorch_input.grad.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_soft_target_cross_entropy_graph(dim: int):
    (
        pytorch_input,
        oneflow_input,
        pytorch_target,
        oneflow_target,
    ) = generate_necessity_for_soft_target_cross_entropy(4)

    class CurrentGraph(flow.nn.Graph):
        def __init__(self) -> None:
            super().__init__()
            self.f = flow.nn.SoftTargetCrossEntropy()

        def build(self, x, y):
            return self.f(x, y)

    pytorch_out = SoftTargetCrossEntropy()(pytorch_input, pytorch_target)
    oneflow_out = CurrentGraph()(oneflow_input, oneflow_target)
    assert np.allclose(
        oneflow_out.numpy(), pytorch_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )


@flow.unittest.skip_unless_1n1d()
class TestSoftTargetCrossEntropy(flow.unittest.TestCase):
    def test_soft_target_cross_entropy(test_case):
        for i in range(10):
            test_soft_target_cross_entropy(2)
            test_soft_target_cross_entropy(3)
            test_soft_target_cross_entropy(4)
            test_soft_target_cross_entropy(5)
            test_soft_target_cross_entropy_graph(2)
            test_soft_target_cross_entropy_graph(3)
            test_soft_target_cross_entropy_graph(4)
            test_soft_target_cross_entropy_graph(5)


class JsdCrossEntropy(pytorch.nn.Module):
    def __init__(self, num_splits=3, alpha=12):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        # if smoothing is not None and smoothing > 0:
        #     self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        # else:
        #     self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.cross_entropy_loss = pytorch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = pytorch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [
            pytorch.nn.functional.softmax(logits, dim=1) for logits in logits_split
        ]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = pytorch.clamp(pytorch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss += (
            self.alpha
            * sum(
                [
                    pytorch.nn.functional.kl_div(
                        logp_mixture, p_split, reduction="batchmean"
                    )
                    for p_split in probs
                ]
            )
            / len(probs)
        )
        return loss


def generate_necessity_for_jsd_cross_entropy(dim: int):
    if dim > 5 or dim < 2:
        raise ValueError("dim should be less than 5 or greater than 1. ")
    device = random_device()
    num_classes = random(low=2).to(int)
    batch_size = 24
    extra_dim = [random().to(int) for _ in range(dim - 2)]

    def get_tensor_tuple(obtained_random_tensor, requires_grad=True):
        pytorch_tensor = obtained_random_tensor.value().requires_grad_(requires_grad)
        flow_tensor = flow.tensor(
            pytorch_tensor.detach().cpu().numpy(), requires_grad=requires_grad,
        )
        return pytorch_tensor, flow_tensor

    return (
        *get_tensor_tuple(
            random_tensor(dim, batch_size, num_classes, *extra_dim).to(device)
        ),
        *get_tensor_tuple(
            random_tensor(
                dim - 1, batch_size, *extra_dim, low=0, high=num_classes, dtype=int,
            ).to(device),
            False,
        ),
    )


def test_jsd_cross_entropy(dim: int):
    (
        pytorch_input,
        oneflow_input,
        pytorch_target,
        oneflow_target,
    ) = generate_necessity_for_jsd_cross_entropy(dim)
    pytorch_out = JsdCrossEntropy()(pytorch_input, pytorch_target)
    pytorch_out.sum().backward()
    oneflow_out = flow.nn.JsdCrossEntropy()(oneflow_input, oneflow_target)
    oneflow_out.sum().backward()
    assert np.allclose(
        oneflow_out.numpy(), pytorch_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    assert np.allclose(
        oneflow_input.grad.numpy(),
        pytorch_input.grad.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


@flow.unittest.skip_unless_1n1d()
class TestJsdCrossEntropy(flow.unittest.TestCase):
    def test_jsd_cross_entropy(test_case):
        for i in range(10):
            test_jsd_cross_entropy(2)
            test_jsd_cross_entropy(3)
            test_jsd_cross_entropy(4)
            test_jsd_cross_entropy(5)


if __name__ == "__main__":
    unittest.main()
