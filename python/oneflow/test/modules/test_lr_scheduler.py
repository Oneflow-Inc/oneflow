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

import math
import random
import tempfile
import unittest
import numpy as np
from collections import OrderedDict

import oneflow as flow
import oneflow.unittest
import torch
from oneflow.nn.parameter import Parameter

from oneflow.test_utils.test_util import GenArgDict


def compare_with_torch_reduce_lr(
    test_case, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps,
):
    optimizer_flow = flow.optim.SGD(
        [{"params": [Parameter(flow.Tensor([1.0]))]},],
        lr=TestLrScheduler.base_lr,
        momentum=0.9,
    )

    optimizer_torch = torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.Tensor([1.0]))]},],
        lr=TestLrScheduler.base_lr,
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


@flow.unittest.skip_unless_1n1d()
class TestLrScheduler(flow.unittest.TestCase):
    base_lr = 1.0

    def test_cosine_decay_lr(test_case):
        optimizer = flow.optim.SGD(
            [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=TestLrScheduler.base_lr
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
            new_lr = cosine_decay_lr_step(
                TestLrScheduler.base_lr, i, decay_steps, alpha
            )
            test_case.assertAlmostEqual(
                cosine_decay_lr.get_last_lr()[0], new_lr, places=4
            )

    def test_cosine_annealing_lr(test_case):
        optimizer = flow.optim.SGD(
            [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=TestLrScheduler.base_lr
        )

        def cosine_annealing_lr_step(base_lr, current_step, last_lr, T_max, eta_min):
            if (current_step - 1 - T_max) % (2 * T_max) == 0:
                return (
                    last_lr
                    + (TestLrScheduler.base_lr - eta_min)
                    * (1 - math.cos(math.pi / T_max))
                    / 2
                )
            else:
                return (1 + math.cos(math.pi * current_step / T_max)) / (
                    1 + math.cos(math.pi * (current_step - 1) / T_max)
                ) * (last_lr - eta_min) + eta_min

        T_max = 20
        eta_min = 0.5
        cosine_annealing_lr = flow.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
        numpy_last_lr = TestLrScheduler.base_lr
        for i in range(1, 101):
            cosine_annealing_lr.step()
            numpy_last_lr = cosine_annealing_lr_step(
                TestLrScheduler.base_lr, i, numpy_last_lr, T_max, eta_min
            )
            test_case.assertAlmostEqual(
                cosine_annealing_lr.get_last_lr()[0], numpy_last_lr, places=4
            )

    def test_step_lr(test_case):
        optimizer = flow.optim.SGD(
            [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=TestLrScheduler.base_lr
        )

        def step_lr_step(base_lr, current_step, step_size, gamma):
            return base_lr * gamma ** (current_step // step_size)

        gamma = 0.1
        step_size = 5
        step_lr = flow.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        for i in range(1, 21):
            step_lr.step()
            new_lr = step_lr_step(TestLrScheduler.base_lr, i, step_size, gamma)
            test_case.assertAlmostEqual(step_lr.get_last_lr()[0], new_lr, places=5)

    def test_multistep_lr(test_case):
        optimizer = flow.optim.SGD(
            [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=TestLrScheduler.base_lr
        )

        def multistep_lr_step(base_lr, current_step, milestones, gamma):
            count = 0
            for step in milestones:
                if current_step >= step:
                    count += 1
            return base_lr * gamma ** count

        gamma = 0.1
        milestones = [5, 11, 15]
        multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
        for i in range(1, 18):
            multistep_lr.step()
            new_lr = multistep_lr_step(TestLrScheduler.base_lr, i, milestones, gamma)
            test_case.assertAlmostEqual(multistep_lr.get_last_lr()[0], new_lr, places=5)

    def test_exponential_lr(test_case):
        optimizer = flow.optim.SGD(
            [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=TestLrScheduler.base_lr
        )

        def exponential_lr_step(base_lr, current_step, gamma):
            return base_lr * gamma ** current_step

        gamma = 0.1
        exponential_lr = flow.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        for i in range(1, 21):
            exponential_lr.step()
            new_lr = exponential_lr_step(TestLrScheduler.base_lr, i, gamma)
            test_case.assertAlmostEqual(
                exponential_lr.get_last_lr()[0], new_lr, places=5
            )

    def test_lambda_lr(test_case):
        optimizer = flow.optim.SGD(
            [
                {"params": [Parameter(flow.Tensor([1.0]))]},
                {"params": [Parameter(flow.Tensor([1.0]))]},
            ],
            lr=TestLrScheduler.base_lr,
        )
        lambdas = [lambda step: step // 30, lambda step: 0.95 * step]

        def lambda_lr_step(base_lrs, current_step):
            return [
                base_lr * lmbda(current_step)
                for (base_lr, lmbda) in zip(base_lrs, lambdas)
            ]

        lambda_lr = flow.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        for i in range(1, 21):
            lambda_lr.step()
            new_lrs = lambda_lr_step(lambda_lr.base_lrs, i)
            for (lr1, lr2) in zip(lambda_lr.get_last_lr(), new_lrs):
                test_case.assertAlmostEqual(lr1, lr2, places=5)

    def test_polynomial_lr(test_case):
        optimizer = flow.optim.SGD(
            [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=TestLrScheduler.base_lr
        )

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
        poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
            optimizer, decay_steps, end_learning_rate, power, cycle
        )
        # step(0) will be invoked in LRScheduler.__init__
        new_lr = polynomial_lr_step(
            TestLrScheduler.base_lr, end_learning_rate, 0, decay_steps, power, cycle
        )
        test_case.assertAlmostEqual(poly_decay_lr.get_last_lr()[0], new_lr, places=4)
        for i in range(1, 21):
            poly_decay_lr.step()
            new_lr = polynomial_lr_step(
                TestLrScheduler.base_lr, end_learning_rate, i, decay_steps, power, cycle
            )
            test_case.assertAlmostEqual(
                poly_decay_lr.get_last_lr()[0], new_lr, places=4
            )

        cycle = True
        poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
            optimizer, decay_steps, end_learning_rate, power, cycle
        )
        for i in range(1, 21):
            poly_decay_lr.step()
            new_lr = polynomial_lr_step(
                TestLrScheduler.base_lr, end_learning_rate, i, decay_steps, power, cycle
            )
            test_case.assertAlmostEqual(
                poly_decay_lr.get_last_lr()[0], new_lr, places=4
            )

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
            compare_with_torch_reduce_lr(test_case, **arg)

    def test_warmup_scheduler_save_and_load(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))

        optimizer = flow.optim.SGD([param])
        cosine_scheduler = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
            cosine_scheduler, warmup_factor=0.1, warmup_iters=5, warmup_method="linear",
        )
        for _ in range(random.randint(1, 10)):
            lr_scheduler.step()
        # save
        with tempfile.NamedTemporaryFile() as f:
            flow.save(lr_scheduler.state_dict(), f.name)
            state_dict = flow.load(f.name)

        # load
        param2 = flow.nn.Parameter(flow.ones(3, 4))
        optimizer2 = flow.optim.SGD([param])
        cosine_scheduler2 = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
        lr_scheduler2 = flow.optim.lr_scheduler.WarmUpLR(
            cosine_scheduler2,
            warmup_factor=0.5,
            warmup_iters=10,
            warmup_method="linear",
        )
        lr_scheduler2.load_state_dict(state_dict)

        # compare warm up scheduler
        for attr in ["warmup_iters", "warmup_factor", "warmup_method", "last_step"]:
            test_case.assertEqual(
                getattr(lr_scheduler, attr), getattr(lr_scheduler2, attr)
            )
        # compare cosine_annealing_lr
        for attr in ["T_max", "eta_min", "last_step"]:
            test_case.assertEqual(
                getattr(cosine_scheduler, attr), getattr(cosine_scheduler2, attr)
            )


@flow.unittest.skip_unless_1n1d()
class WarmupLRTestCase(flow.unittest.TestCase):
    def test_only_warmup(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.001)
        warmup_lr = flow.optim.lr_scheduler.WarmupLR(
            optimizer, warmup_factor=0.5, warmup_iters=5, warmup_method="linear"
        )
        expected_lrs = [
            0.0005,
            0.0006,
            0.0007,
            0.0008,
            0.0009,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
        ]
        lrs = [warmup_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            optimizer.step()
            warmup_lr.step()
            lrs.append(warmup_lr.get_last_lr()[0])

        lrs = lrs[:-1]

        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_warmup_iters_0_exp_lr(test_case):
        lr = 0.1
        gamma = 0.9
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr)
        exp_lr = flow.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        warmup_lr = flow.optim.lr_scheduler.WarmupLR(
            exp_lr, warmup_factor=0.5, warmup_iters=0, warmup_method="linear"
        )
        iters = 10
        lrs = [warmup_lr.get_last_lr()[0]]
        for _ in range(iters):
            warmup_lr.step()
            lrs.append(warmup_lr.get_last_lr()[0])

        lrs = lrs[:-1]
        expected_lrs = [lr * pow(gamma, i) for i in range(iters)]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_linear_warmup_exp_lr(test_case):
        lr = 0.1
        gamma = 0.9
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr)
        exp_lr = flow.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        warmup_lr = flow.optim.lr_scheduler.WarmupLR(
            exp_lr, warmup_factor=0.5, warmup_iters=5, warmup_method="linear"
        )
        expected_lrs = [
            0.05,
            0.0518098,
            0.0536196,
            0.0554294,
            0.0572392,
            0.059049,
            0.0531441,
            0.04782969,
            0.043046721,
            0.0387420489,
        ]

        lrs = [warmup_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            warmup_lr.step()
            lrs.append(warmup_lr.get_last_lr()[0])

        lrs = lrs[:-1]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_linear_warmup_prefix_exp_lr(test_case):
        lr = 0.1
        gamma = 0.9
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr)
        exp_lr = flow.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        warmup_lr = flow.optim.lr_scheduler.WarmupLR(
            exp_lr,
            warmup_factor=0.5,
            warmup_iters=5,
            warmup_method="linear",
            warmup_prefix=True,
        )
        expected_lrs = [
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.1,
            0.09,
            0.081,
            0.0729,
            0.06561,
        ]

        lrs = [warmup_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            warmup_lr.step()
            lrs.append(warmup_lr.get_last_lr()[0])

        lrs = lrs[:-1]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_constant_warmup_cosine_annealing(test_case):
        lr = 0.1
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr)
        cos_annl_lr = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        warmup_lr = flow.optim.lr_scheduler.WarmupLR(
            cos_annl_lr, warmup_factor=0.5, warmup_iters=5, warmup_method="constant",
        )

        expected_lrs = [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.03454915028125264,
            0.020610737385376353,
            0.009549150281252635,
            0.002447174185242324,
            0.0,
            0.0024471741852423235,
            0.009549150281252666,
            0.020610737385376433,
            0.034549150281252786,
            0.050000000000000225,
            0.06545084971874766,
            0.079389262614624,
            0.09045084971874778,
            0.09755282581475812,
            0.1,
        ]

        lrs = [warmup_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            warmup_lr.step()
            lrs.append(warmup_lr.get_last_lr()[0])

        lrs = lrs[:-1]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_linear_warmup_cosine_annealing(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.1)
        cos_annl_lr = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        warmup_lr = flow.optim.lr_scheduler.WarmupLR(
            cos_annl_lr, warmup_factor=0.1, warmup_iters=5, warmup_method="linear",
        )

        expected_lrs = [
            0.01,
            0.025071068,
            0.040142136,
            0.055213203,
            0.070284271,
            0.085355339,
            0.079389263,
            0.072699525,
            0.06545085,
            0.057821723,
            0.05,
            0.042178277,
            0.03454915,
            0.027300475,
            0.020610737,
            0.014644661,
            0.00954915,
            0.005449674,
            0.002447174,
            0.000615583,
        ]

        lrs = [warmup_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            warmup_lr.step()
            lrs.append(warmup_lr.get_last_lr()[0])

        lrs = lrs[:-1]

        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_linear_warmup_prefix_cosine_annealing(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.1)
        cos_annl_lr = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        warmup_lr = flow.optim.lr_scheduler.WarmupLR(
            cos_annl_lr,
            warmup_factor=0.1,
            warmup_iters=5,
            warmup_method="linear",
            warmup_prefix=True,
        )

        expected_lrs = [
            0.01,
            0.028,
            0.046,
            0.064,
            0.082,
            0.1,
            0.099384417,
            0.097552826,
            0.094550326,
            0.09045085,
            0.085355339,
            0.079389263,
            0.072699525,
            0.06545085,
            0.057821723,
            0.05,
            0.042178277,
            0.03454915,
            0.027300475,
            0.020610737,
        ]

        lrs = [warmup_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            warmup_lr.step()
            lrs.append(warmup_lr.get_last_lr()[0])

        lrs = lrs[:-1]

        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_linear_warmup_multistep_lr(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.001)
        multistep_lr = flow.optim.lr_scheduler.MultiStepLR(optimizer, [10])
        warmup_lr = flow.optim.lr_scheduler.WarmupLR(
            multistep_lr, warmup_factor=0.5, warmup_iters=5, warmup_method="linear",
        )
        expected_lrs = [
            0.0005,
            0.0006,
            0.0007,
            0.0008,
            0.0009,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
        ]
        lrs = [warmup_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            optimizer.step()
            warmup_lr.step()
            lrs.append(warmup_lr.get_last_lr()[0])

        lrs = lrs[:-1]

        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_linear_warmup_prefix_multistep_lr(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.1)
        multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 10]
        )
        warmup_lr = flow.optim.lr_scheduler.WarmupLR(
            multistep_lr,
            warmup_factor=0.1,
            warmup_iters=5,
            warmup_method="linear",
            warmup_prefix=True,
        )

        expected_lrs = [
            0.01,
            0.028,
            0.046,
            0.064,
            0.082,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
        ]

        lrs = [warmup_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            warmup_lr.step()
            lrs.append(warmup_lr.get_last_lr()[0])

        lrs = lrs[:-1]

        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )


@flow.unittest.skip_unless_1n1d()
class ConstantLRTestCase(flow.unittest.TestCase):
    def test(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.01)
        constant_lr = flow.optim.lr_scheduler.ConstantLR(optimizer, 0.1, 10)
        expected_lrs = [
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ]
        lrs = [constant_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            constant_lr.step()
            lrs.append(constant_lr.get_last_lr()[0])

        lrs = lrs[:-1]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )


@flow.unittest.skip_unless_1n1d()
class LinearLRTestCase(flow.unittest.TestCase):
    def test(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.1)
        linear_lr = flow.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1, 10)
        expected_lrs = [
            0.01,
            0.019,
            0.028,
            0.037,
            0.046,
            0.055,
            0.064,
            0.073,
            0.082,
            0.091,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ]
        lrs = [linear_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            linear_lr.step()
            lrs.append(linear_lr.get_last_lr()[0])

        lrs = lrs[:-1]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_end_factor(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.1)
        linear_lr = flow.optim.lr_scheduler.LinearLR(optimizer, 0.1, 0.9, 10)
        expected_lrs = [
            0.01,
            0.018,
            0.026,
            0.034,
            0.042,
            0.05,
            0.058,
            0.066,
            0.074,
            0.082,
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,
        ]
        lrs = [linear_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            linear_lr.step()
            lrs.append(linear_lr.get_last_lr()[0])

        lrs = lrs[:-1]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )


@flow.unittest.skip_unless_1n1d()
class ChainedSchedulerTestCase(flow.unittest.TestCase):
    def test(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        opt = flow.optim.SGD([param], lr=1)
        s1 = flow.optim.lr_scheduler.ConstantLR(opt, factor=0.1, total_iters=3)
        s2 = flow.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        scheduler = flow.optim.lr_scheduler.ChainedScheduler([s1, s2])

        expected_lrs = [0.1, 0.09, 0.081, 0.729, 0.6561, 0.59049]
        lrs = [scheduler.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        lrs = lrs[: len(expected_lrs)]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )


@flow.unittest.skip_unless_1n1d()
class CosineAnnealingWarmRestartsTestCase(flow.unittest.TestCase):
    def test_mult_1(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.1)
        cosa_r_lr = flow.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, eta_min=0.01,
        )
        # fmt: off
        expected_lrs = [0.1, 0.09779754323328192, 0.09140576474687263, 0.08145033635316129, 0.06890576474687264, 0.05500000000000001, 0.04109423525312737, 0.028549663646838717, 0.01859423525312737, 0.012202456766718092, 0.1, 0.09779754323328192, 0.09140576474687263, 0.08145033635316129, 0.06890576474687264, 0.05500000000000001, 0.04109423525312737, 0.028549663646838717, 0.01859423525312737, 0.012202456766718092, 0.1, 0.09779754323328192, 0.09140576474687263, 0.08145033635316129, 0.06890576474687264, 0.05500000000000001, 0.04109423525312737, 0.028549663646838717, 0.01859423525312737, 0.012202456766718092, 0.1, 0.09779754323328192, 0.09140576474687263, 0.08145033635316129, 0.06890576474687264, 0.05500000000000001, 0.04109423525312737, 0.028549663646838717, 0.01859423525312737, 0.012202456766718092, 0.1, 0.09779754323328192, 0.09140576474687263, 0.08145033635316129, 0.06890576474687264, 0.05500000000000001, 0.04109423525312737, 0.028549663646838717, 0.01859423525312737, 0.012202456766718092]
        # fmt: on
        lrs = [cosa_r_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            cosa_r_lr.step()
            lrs.append(cosa_r_lr.get_last_lr()[0])

        lrs = lrs[: len(expected_lrs)]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_mult_2(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.1)
        cosa_r_lr = flow.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0.01,
        )
        # fmt: off
        expected_lrs = [0.1, 0.09779754323328192, 0.09140576474687263, 0.08145033635316129, 0.06890576474687264, 0.05500000000000001, 0.04109423525312737, 0.028549663646838717, 0.01859423525312737, 0.012202456766718092, 0.1, 0.0994459753267812, 0.09779754323328192, 0.09509529358847656, 0.09140576474687263, 0.08681980515339464, 0.08145033635316129, 0.07542957248827961, 0.06890576474687264, 0.0620395509268104, 0.05500000000000001, 0.04796044907318963, 0.04109423525312737, 0.034570427511720396, 0.028549663646838717, 0.023180194846605363, 0.01859423525312737, 0.014904706411523451, 0.012202456766718092, 0.010554024673218806, 0.1, 0.09986128001799077, 0.0994459753267812, 0.09875664641789544, 0.09779754323328192, 0.0965745789630079, 0.09509529358847656, 0.09336880739593416, 0.09140576474687263, 0.0892182684520014, 0.08681980515339464, 0.08422516217485827, 0.08145033635316129, 0.0785124354122177, 0.07542957248827961, 0.07222075445642905, 0.06890576474687264, 0.06550504137351576, 0.0620395509268104, 0.05853065930775304, 0.05500000000000001, 0.05146934069224699, 0.04796044907318963, 0.04449495862648427, 0.04109423525312737, 0.03777924554357097, 0.034570427511720396, 0.031487564587782305, 0.028549663646838717, 0.02577483782514174, 0.023180194846605363, 0.02078173154799861, 0.01859423525312737, 0.016631192604065852, 0.014904706411523451, 0.013425421036992097, 0.012202456766718092, 0.011243353582104555, 0.010554024673218806, 0.010138719982009242]
        # fmt: on
        lrs = [cosa_r_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            cosa_r_lr.step()
            lrs.append(cosa_r_lr.get_last_lr()[0])

        lrs = lrs[: len(expected_lrs)]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )

    def test_mult_2_decay_half_limit_2(test_case):
        param = flow.nn.Parameter(flow.ones(3, 4))
        optimizer = flow.optim.SGD([param], lr=0.1)
        cosa_r_lr = flow.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, decay_rate=0.5, restart_limit=2, eta_min=0.01,
        )
        # fmt: off
        expected_lrs = [0.1, 0.09779754323328192, 0.09140576474687263, 0.08145033635316129, 0.06890576474687264, 0.05500000000000001, 0.04109423525312737, 0.028549663646838717, 0.01859423525312737, 0.012202456766718092, 0.05, 0.04975376681190276, 0.04902113032590308, 0.04782013048376736, 0.04618033988749895, 0.044142135623730955, 0.04175570504584947, 0.03907980999479094, 0.03618033988749895, 0.03312868930080462, 0.03, 0.02687131069919539, 0.023819660112501053, 0.020920190005209068, 0.018244294954150538, 0.01585786437626905, 0.013819660112501053, 0.012179869516232645, 0.01097886967409693, 0.010246233188097247, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        # fmt: on
        lrs = [cosa_r_lr.get_last_lr()[0]]
        for _ in range(len(expected_lrs)):
            cosa_r_lr.step()
            lrs.append(cosa_r_lr.get_last_lr()[0])

        lrs = lrs[: len(expected_lrs)]
        test_case.assertTrue(
            np.allclose(lrs, expected_lrs),
            f"\nexpected_lrs: {expected_lrs}\nvs.\ncalculated lrs: {lrs}",
        )


if __name__ == "__main__":
    unittest.main()
