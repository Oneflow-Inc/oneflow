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
from oneflow.test_utils.automated_test_util import *


def _assert_true(test_case, value1, value2, name=""):
    is_equal = np.allclose(
        value1.detach().cpu().numpy(), value2.detach().numpy(), rtol=1e-03, atol=1e-03,
    )
    test_case.assertTrue(is_equal, f"{name} is not equal." if name else "")


def generate_grads_for_variables(variables):
    if isinstance(variables, list):
        shape_and_sbp = [(i.oneflow.shape, i.oneflow.sbp) for i in variables]
        placement = variables[0].oneflow.placement
    elif hasattr(variables, "pytorch"):
        shape_and_sbp = [(i.shape, i.sbp) for i in variables.oneflow]
        placement = variables.oneflow[0].placement
    else:
        assert False
    grads = [
        random_tensor(
            len(shape), *shape, requires_grad=random_bool().value()
        ).to_global(placement=placement, sbp=sbp)
        for shape, sbp in shape_and_sbp
    ]
    return grads


def calculate_and_compare_loss(test_case, input, target, model, order=2):
    output = model(input, target)
    _assert_true(test_case, output.pytorch, output.oneflow, "output")

    init_inputs = [input, target]
    grad_inputs = [output]
    grad_outputs = []
    for i in range(order):
        inputs = [
            var for var in [*init_inputs, *grad_outputs] if var.pytorch.requires_grad
        ]
        outputs = grad_inputs
        grad_outputs = generate_grads_for_variables(outputs)
        if i == order - 1:
            grad_inputs = torch.autograd.grad(outputs, inputs, grad_outputs)
        else:
            grad_inputs = torch.autograd.grad(outputs, inputs, grad_outputs, True, True)
        for j in range(len(inputs)):
            _assert_true(
                test_case,
                grad_inputs[j].pytorch,
                grad_inputs[j].oneflow,
                f"{i}-grad_inputs[{j}]",
            )


def generate_necessity_for_default_loss(placement):
    shape = [8, 8]
    ndim = len(shape)
    input_requires_grad = True
    target_requires_grad = random_bool().value()
    return (
        random_tensor(ndim, *shape, low=0, requires_grad=input_requires_grad).to_global(
            placement=placement, sbp=random_sbp(placement, max_dim=2)
        ),
        random_tensor(
            ndim, *shape, low=0, requires_grad=target_requires_grad
        ).to_global(placement=placement, sbp=random_sbp(placement, max_dim=2)),
    )


def generate_necessity_for_nll_loss(placement):
    ndim = 2
    num_classes = 8
    batch_size = 8
    ignore_index = oneof(random(0, num_classes).to(int).value(), -100).value()
    extra_dim = [random().to(int) for _ in range(ndim - 2)]
    return (
        random_tensor(ndim, batch_size, num_classes).to_global(
            placement=placement, sbp=random_sbp(placement, max_dim=2)
        ),
        random_tensor(
            ndim - 1,
            batch_size,
            low=0,
            high=num_classes,
            dtype=int,
            requires_grad=False,
        ).to_global(placement=placement, sbp=random_sbp(placement, max_dim=1)),
        random_tensor(1, num_classes, low=0, high=3, requires_grad=False).to_global(
            placement=placement, sbp=random_sbp(placement, except_split=True)
        ),
        ignore_index,
    )


def generate_necessity_for_bce_loss(placement):
    ndim = 3
    num_classes = 2
    batch_size = 8
    extra_dim = [random().to(int) for _ in range(ndim - 2)]
    input_requires_grad = True
    target_requires_grad = False
    return (
        random_tensor(
            ndim,
            batch_size,
            num_classes,
            low=0,
            high=1,
            *extra_dim,
            requires_grad=input_requires_grad,
        ).to_global(placement=placement, sbp=random_sbp(placement, max_dim=1)),
        random_tensor(
            ndim,
            batch_size,
            num_classes,
            *extra_dim,
            low=0,
            high=num_classes,
            requires_grad=target_requires_grad,
        ).to_global(placement=placement, sbp=random_sbp(placement, max_dim=1)),
        random_tensor(
            ndim,
            batch_size,
            num_classes,
            *extra_dim,
            low=0,
            high=3,
            requires_grad=False,
        ).to_global(placement=placement, sbp=random_sbp(placement, max_dim=1)),
        random_tensor(1, 1, low=1, high=3, requires_grad=False,).to_global(
            placement=placement, sbp=random_sbp(placement, except_split=True)
        ),
    )


def _test_smooth_l1_loss_grad_grad_impl(test_case, placement):
    x, y = generate_necessity_for_default_loss(placement)

    m = torch.nn.SmoothL1Loss(
        reduction=oneof("none", "sum", "mean", nothing()), beta=oneof(0.0, 0.5, 1)
    )

    calculate_and_compare_loss(test_case, x, y, m)


def _test_kl_div_loss_grad_grad_impl(test_case, placement):
    x, y = generate_necessity_for_default_loss(placement)

    m = torch.nn.KLDivLoss(
        reduction=oneof("none", "sum", "mean", nothing()),
        log_target=oneof(True, False),
    )

    calculate_and_compare_loss(test_case, x, y, m)


def _test_bce_loss_grad_grad_impl(test_case, placement, with_logits=False):
    x, y, weight, pos_weight = generate_necessity_for_bce_loss(placement)

    if with_logits:
        weight = weight if random_bool().value() else None
        has_pos_weight = random_bool().value()
        pos_weight = pos_weight if has_pos_weight else nothing()
        m = torch.nn.BCEWithLogitsLoss(
            weight=weight,
            pos_weight=pos_weight,
            reduction=oneof("none", "sum", "mean"),
        )
        if has_pos_weight:
            y = y.detach().clone().requires_grad_(False)
    else:
        m = torch.nn.BCELoss(
            weight=(weight if random_bool().value() else None),
            reduction=oneof("none", "sum", "mean"),
        )

    calculate_and_compare_loss(test_case, x, y, m)


def _test_nll_loss_grad_grad_impl(test_case, placement):
    (x, y, weight, ignore_index) = generate_necessity_for_nll_loss(placement)

    m = torch.nn.NLLLoss(
        weight=(weight if random_bool().value() else None),
        reduction=oneof("none", "sum", "mean", nothing()),
        ignore_index=ignore_index,
    )

    calculate_and_compare_loss(test_case, x, y, m)


class TestGlobalLossHigherDerivative(flow.unittest.TestCase):
    @globaltest
    def test_smooth_l1_loss_grad_grad(test_case):
        for placement in all_placement():
            _test_smooth_l1_loss_grad_grad_impl(test_case, placement)

    @globaltest
    def test_kl_div_loss_grad_grad(test_case):
        for placement in all_placement():
            _test_kl_div_loss_grad_grad_impl(test_case, placement)

    @globaltest
    def test_nll_loss_grad_grad(test_case):
        for placement in all_placement():
            _test_nll_loss_grad_grad_impl(test_case, placement)

    @globaltest
    def test_bce_loss_grad_grad(test_case):
        for placement in all_placement():
            _test_bce_loss_grad_grad_impl(test_case, placement)

    @globaltest
    def test_bce_with_logits_loss_grad_grad(test_case):
        for placement in all_placement():
            _test_bce_loss_grad_grad_impl(test_case, placement, with_logits=True)


if __name__ == "__main__":
    unittest.main()
