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
import math
import torch
import random
import oneflow as flow
import oneflow.unittest

from collections import OrderedDict
from oneflow.nn.parameter import Parameter
from test_util import GenArgDict
from oneflow.test_utils.automated_test_util import all_placement
from oneflow.test_utils.automated_test_util import all_sbp
from oneflow.test_utils.automated_test_util import consistent


def _test_cosine_decay_lr(test_case, placement, sbp, base_lr):
    optimizer = flow.optim.SGD(
        [{"params": [Parameter(flow.Tensor([1.0]).to_consistent(placement, sbp))]}],
        base_lr,
    )

    def cosine_decay_lr_step(base_lr, current_step, decay_steps, alpha):
        if current_step < decay_steps:
            cos_decay = 0.5 * (1 + math.cos(math.pi * current_step / decay_steps))
            decay_factor = (1 - alpha) * cos_decay + alpha
            return base_lr * decay_factor
        else:
            return base_lr * alpha

    alpha = 0.5
    decay_steps = 10
    cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=decay_steps, alpha=alpha
    )
    for i in range(1, 21):
        cosine_decay_lr.step()
        new_lr = cosine_decay_lr_step(base_lr, i, decay_steps, alpha)
        test_case.assertAlmostEqual(cosine_decay_lr.get_last_lr()[0], new_lr, places=4)


def _test_cosine_annealing_lr(test_case, placement, sbp, base_lr):
    T_max = 20
    eta_min = 0.5
    flow_optimizer = flow.optim.SGD(
        [{"params": [Parameter(flow.Tensor([1.0]).to_consistent(placement, sbp))]}],
        lr=base_lr,
    )
    flow_cosine_annealing_lr = flow.optim.lr_scheduler.CosineAnnealingLR(
        flow_optimizer, T_max=T_max, eta_min=eta_min
    )

    torch_optimizer = torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.Tensor([1.0]))]},], lr=base_lr
    )
    torch_cosine_annealing_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        torch_optimizer, T_max=T_max, eta_min=eta_min
    )
    for i in range(1, 101):
        flow_cosine_annealing_lr.step()
        torch_cosine_annealing_lr.step()
        test_case.assertAlmostEqual(
            flow_cosine_annealing_lr.get_last_lr()[0],
            torch_cosine_annealing_lr.get_last_lr()[0],
        )


def _test_step_lr(test_case, placement, sbp, base_lr):
    gamma = 0.1
    step_size = 5
    flow_optimizer = flow.optim.SGD(
        [{"params": [Parameter(flow.Tensor([1.0]).to_consistent(placement, sbp))]}],
        lr=base_lr,
    )
    flow_step_lr = flow.optim.lr_scheduler.StepLR(
        flow_optimizer, step_size=step_size, gamma=gamma
    )

    torch_optimizer = torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.Tensor([1.0]))]},], lr=base_lr
    )
    torch_step_lr = torch.optim.lr_scheduler.StepLR(
        torch_optimizer, step_size=step_size, gamma=gamma
    )
    for i in range(1, 21):
        flow_step_lr.step()
        torch_step_lr.step()
        test_case.assertAlmostEqual(
            flow_step_lr.get_last_lr()[0], torch_step_lr.get_last_lr()[0], places=5
        )


def _test_multistep_lr(test_case, placement, sbp, base_lr):
    gamma = 0.1
    milestones = [5, 11, 15]
    flow_optimizer = flow.optim.SGD(
        [{"params": [Parameter(flow.Tensor([1.0]).to_consistent(placement, sbp))]}],
        lr=base_lr,
    )
    flow_multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        flow_optimizer, milestones=milestones, gamma=gamma
    )
    torch_optimizer = torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.Tensor([1.0]))]},], lr=base_lr
    )
    torch_multistep_lr = torch.optim.lr_scheduler.MultiStepLR(
        torch_optimizer, milestones=milestones, gamma=gamma
    )
    for i in range(1, 18):
        flow_multistep_lr.step()
        torch_multistep_lr.step()
        test_case.assertAlmostEqual(
            flow_multistep_lr.get_last_lr()[0],
            torch_multistep_lr.get_last_lr()[0],
            places=5,
        )


def _test_exponential_lr(test_case, placement, sbp, base_lr):
    gamma = 0.1
    flow_optimizer = flow.optim.SGD(
        [{"params": [Parameter(flow.Tensor([1.0]).to_consistent(placement, sbp))]}],
        lr=base_lr,
    )
    flow_exponential_lr = flow.optim.lr_scheduler.ExponentialLR(
        flow_optimizer, gamma=gamma
    )
    torch_optimizer = torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.Tensor([1.0]))]},], lr=base_lr
    )
    torch_exponential_lr = torch.optim.lr_scheduler.ExponentialLR(
        torch_optimizer, gamma=gamma
    )
    for i in range(1, 18):
        flow_exponential_lr.step()
        torch_exponential_lr.step()
        test_case.assertAlmostEqual(
            flow_exponential_lr.get_last_lr()[0],
            torch_exponential_lr.get_last_lr()[0],
            places=5,
        )


def _test_polynomial_lr(test_case, placement, sbp, base_lr):
    def polynomial_lr_step(base_lr, end_lr, step, decay_steps, power, cycle):
        if cycle:
            if step == 0:
                step = 1
            decay_steps = decay_steps * math.ceil(step / decay_steps)
        step = min(step, decay_steps)
        return (base_lr - end_lr) * (1 - step / decay_steps) ** power + end_lr

    decay_steps = 100
    end_learning_rate = 1e-5
    power = 2
    cycle = True
    optimizer = flow.optim.SGD(
        [{"params": [Parameter(flow.Tensor([1.0]).to_consistent(placement, sbp))]}],
        lr=base_lr,
    )
    poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer, decay_steps, end_learning_rate, power, cycle
    )

    # step(0) will be invoked in LrScheduler.__init__
    new_lr = polynomial_lr_step(
        TestLrScheduler.base_lr, end_learning_rate, 0, decay_steps, power, cycle
    )
    test_case.assertAlmostEqual(poly_decay_lr.get_last_lr()[0], new_lr, places=4)
    for i in range(1, 18):
        poly_decay_lr.step()
        new_lr = polynomial_lr_step(
            TestLrScheduler.base_lr, end_learning_rate, i, decay_steps, power, cycle
        )
        test_case.assertAlmostEqual(poly_decay_lr.get_last_lr()[0], new_lr, places=4)

    cycle = True
    poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer, decay_steps, end_learning_rate, power, cycle
    )
    for i in range(1, 21):
        poly_decay_lr.step()
        new_lr = polynomial_lr_step(
            TestLrScheduler.base_lr, end_learning_rate, i, decay_steps, power, cycle
        )
        test_case.assertAlmostEqual(poly_decay_lr.get_last_lr()[0], new_lr, places=4)


def _test_lambda_lr(test_case, placement, sbp, base_lr):
    lambdas = [lambda step: step // 30, lambda step: 0.95 * step]
    flow_optimizer = flow.optim.SGD(
        [
            {"params": [Parameter(flow.Tensor([1.0]).to_consistent(placement, sbp))]},
            {"params": [Parameter(flow.Tensor([1.0]).to_consistent(placement, sbp))]},
        ],
        lr=base_lr,
    )
    flow_lambda_lr = flow.optim.lr_scheduler.LambdaLR(flow_optimizer, lr_lambda=lambdas)

    torch_optimizer = torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.Tensor([1.0]))]},
            {"params": [torch.nn.Parameter(torch.Tensor([1.0]))]},
        ],
        lr=base_lr,
    )
    torch_lambda_lr = torch.optim.lr_scheduler.LambdaLR(
        torch_optimizer, lr_lambda=lambdas
    )
    for i in range(1, 21):
        flow_lambda_lr.step()
        torch_lambda_lr.step()
        test_case.assertAlmostEqual(
            flow_lambda_lr.get_last_lr()[0], torch_lambda_lr.get_last_lr()[0], places=5
        )


def compare_with_torch_reduce_lr(
    test_case,
    placement,
    sbp,
    base_lr,
    mode,
    factor,
    patience,
    threshold,
    threshold_mode,
    cooldown,
    min_lr,
    eps,
):
    optimizer_flow = flow.optim.SGD(
        [{"params": [Parameter(flow.Tensor([1.0]).to_consistent(placement, sbp))]},],
        lr=base_lr,
        momentum=0.9,
    )

    optimizer_torch = torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.Tensor([1.0]))]},],
        lr=base_lr,
        momentum=0.9,
    )

    scheduler_flow = flow.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_flow,
        mode,
        factor,
        patience,
        threshold,
        threshold_mode,
        cooldown,
        min_lr,
        eps,
    )
    scheduler_troch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_torch,
        mode,
        factor,
        patience,
        threshold,
        threshold_mode,
        cooldown,
        min_lr,
        eps,
    )
    val_loss = 0.1
    for epoch in range(15):
        val_loss += (random.random() - 0.5) / 10
        scheduler_flow.step(val_loss)
        scheduler_troch.step(val_loss)
        for (lr1, lr2) in zip(scheduler_flow._last_lr, scheduler_troch._last_lr):
            test_case.assertAlmostEqual(lr1, lr2, places=5)


class TestLrScheduler(flow.unittest.TestCase):
    base_lr = 1.0

    @consistent
    def test_lr_scheduler(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_cosine_decay_lr(
                    test_case, placement, sbp, TestLrScheduler.base_lr
                )
                _test_cosine_annealing_lr(
                    test_case, placement, sbp, TestLrScheduler.base_lr
                )
                _test_step_lr(test_case, placement, sbp, TestLrScheduler.base_lr)
                _test_multistep_lr(test_case, placement, sbp, TestLrScheduler.base_lr)
                _test_exponential_lr(test_case, placement, sbp, TestLrScheduler.base_lr)
                _test_polynomial_lr(test_case, placement, sbp, TestLrScheduler.base_lr)
                _test_lambda_lr(test_case, placement, sbp, TestLrScheduler.base_lr)

    @consistent
    def test_reduce_lr_on_plateau(test_case):
        arg_dict = OrderedDict()
        arg_dict["mode"] = ["min", "max"]
        arg_dict["factor"] = [0.1, 0.3]
        arg_dict["patience"] = [2, 5]
        arg_dict["threshold"] = [1e-3, 1e-5]
        arg_dict["threshold_mode"] = ["rel", "abs"]
        arg_dict["cooldown"] = [0, 1]
        arg_dict["min_lr"] = [0, 1e-3]
        arg_dict["eps"] = [1e-5, 1e-8]
        for arg in GenArgDict(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=1):
                    compare_with_torch_reduce_lr(
                        test_case, placement, sbp, TestLrScheduler.base_lr, **arg
                    )


if __name__ == "__main__":
    unittest.main()
