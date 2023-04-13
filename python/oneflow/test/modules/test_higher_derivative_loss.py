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
        value1.detach().cpu().numpy(), value2.detach().numpy(), rtol=1e-04, atol=1e-04,
    )
    test_case.assertTrue(is_equal, f"{name} is not equal." if name else "")


def generate_grads_for_variables(variables):
    if isinstance(variables, list):
        variables_shape = [i.pytorch.shape for i in variables]
        device = torch.device(str(variables[0].pytorch.device))
    elif hasattr(variables, "pytorch"):
        variables_shape = [i.shape for i in variables.pytorch]
        device = torch.device(str(variables.pytorch[0].device))
    else:
        assert False

    grads = [
        random_tensor(len(shape), *shape, requires_grad=True).to(device)
        for shape in variables_shape
    ]
    return grads


def calculate_and_compare_loss(test_case, input, target, model, order=2):
    output = model(input, target)
    _assert_true(test_case, output.pytorch, output.oneflow)

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


def generate_necessity_for_default_loss():
    ndim = random(2, 6).to(int).value()
    device = random_device()
    shape = [random().to(int) for _ in range(ndim)]
    input_requires_grad = True
    target_requires_grad = random_bool().value()
    return (
        random_tensor(ndim, *shape, requires_grad=input_requires_grad, low=0).to(
            device
        ),
        random_tensor(ndim, *shape, requires_grad=target_requires_grad, low=0).to(
            device
        ),
    )


def generate_necessity_for_nll_loss():
    ndim = random(2, 6).to(int).value()
    device = random_device()
    num_classes = random(low=2).to(int)
    batch_size = random(low=2, high=5).to(int)
    ignore_index = (
        random(0, num_classes).to(int) | nothing()
        if num_classes.value() > 2
        else nothing()
    )
    extra_dim = [random().to(int) for _ in range(ndim - 2)]
    return (
        random_tensor(ndim, batch_size, num_classes, *extra_dim).to(device),
        random_tensor(
            ndim - 1,
            batch_size,
            *extra_dim,
            low=0,
            high=num_classes,
            dtype=int,
            requires_grad=False,
        ).to(device),
        random_tensor(1, num_classes, low=0, high=3, requires_grad=False).to(device),
        ignore_index,
    )


def generate_necessity_for_bce_loss():
    ndim = random(2, 6).to(int).value()
    device = random_device()
    num_classes = 2
    batch_size = random(low=2, high=5).to(int)
    extra_dim = [random().to(int) for _ in range(ndim - 2)]
    input_requires_grad = True
    target_requires_grad = False
    return (
        random_tensor(
            ndim,
            batch_size,
            num_classes,
            *extra_dim,
            requires_grad=input_requires_grad,
            low=0,
            high=1,
        ).to(device),
        random_tensor(
            ndim,
            batch_size,
            num_classes,
            *extra_dim,
            low=0,
            high=num_classes,
            requires_grad=target_requires_grad,
        ).to(device),
        random_tensor(
            ndim,
            batch_size,
            num_classes,
            *extra_dim,
            low=0,
            high=3,
            requires_grad=False,
        ).to(device),
        random_tensor(
            1,
            oneof(extra_dim[-1] if ndim > 2 else num_classes, 1).value(),
            low=1,
            high=3,
            requires_grad=False,
        ).to(device),
    )


def _test_smooth_l1_loss_grad_grad_impl(test_case):
    x, y = generate_necessity_for_default_loss()

    m = torch.nn.SmoothL1Loss(
        reduction=oneof("none", "sum", "mean", nothing()), beta=oneof(0.0, 0.5, 1)
    )
    m.to(x.device)

    calculate_and_compare_loss(test_case, x, y, m)


def _test_kl_div_loss_grad_grad_impl(test_case):
    x, y = generate_necessity_for_default_loss()

    m = torch.nn.KLDivLoss(
        reduction=oneof("none", "sum", "mean", nothing()),
        log_target=oneof(True, False),
    )
    m.to(x.device)

    calculate_and_compare_loss(test_case, x, y, m)


def _test_bce_loss_grad_grad_impl(test_case, with_logits=False):
    x, y, weight, pos_weight = generate_necessity_for_bce_loss()

    if with_logits:
        weight = oneof(weight, nothing())
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
            weight=oneof(weight, nothing()), reduction=oneof("none", "sum", "mean"),
        )
    m.to(x.device)

    calculate_and_compare_loss(test_case, x, y, m)


def _test_nll_loss_grad_grad_impl(test_case):
    (x, y, weight, ignore_index) = generate_necessity_for_nll_loss()
    m = torch.nn.NLLLoss(
        weight=oneof(weight, nothing()),
        reduction=oneof("none", "sum", "mean"),
        ignore_index=ignore_index,
    )
    m.to(x.device)

    calculate_and_compare_loss(test_case, x, y, m)


@flow.unittest.skip_unless_1n1d()
class TestLossHigherDerivative(flow.unittest.TestCase):
    def test_smooth_l1_loss_grad_grad(test_case):
        for i in range(5):
            _test_smooth_l1_loss_grad_grad_impl(test_case)

    def test_kl_div_loss_grad_grad(test_case):
        for i in range(5):
            _test_kl_div_loss_grad_grad_impl(test_case)

    def test_nll_loss_grad_grad(test_case):
        for i in range(5):
            _test_nll_loss_grad_grad_impl(test_case)

    def test_bce_loss_grad_grad(test_case):
        for i in range(5):
            _test_bce_loss_grad_grad_impl(test_case)

    def test_bce_with_logits_loss_grad_grad(test_case):
        for i in range(5):
            _test_bce_loss_grad_grad_impl(test_case, with_logits=True)


if __name__ == "__main__":
    unittest.main()
