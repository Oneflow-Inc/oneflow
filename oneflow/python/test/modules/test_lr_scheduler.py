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
import unittest

import oneflow.experimental as flow
from oneflow.python.nn.parameter import Parameter


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLrScheduler(flow.unittest.TestCase):
    base_lr = 1.0

    def test_cosine_scheduler(test_case):
        optimizer = flow.optim.SGD(
            [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=TestLrScheduler.base_lr
        )

        def cosine_scheduler_step(base_lr, current_step, steps, alpha):
            if current_step < steps:
                cos_decay = 0.5 * (1 + math.cos(math.pi * current_step / steps))
                decay_factor = (1 - alpha) * cos_decay + alpha
                return base_lr * decay_factor
            else:
                return base_lr * alpha

        alpha = 0.5
        steps = 10
        cosine_scheduler = flow.optim.lr_scheduler.CosineScheduler(
            optimizer, steps=steps, alpha=alpha
        )

        for i in range(1, 21):
            cosine_scheduler.step()
            new_lr = cosine_scheduler_step(TestLrScheduler.base_lr, i, steps, alpha)
            test_case.assertAlmostEqual(
                cosine_scheduler.get_last_lr()[0], new_lr, places=4
            )

    def test_step_lr(test_case):
        optimizer = flow.optim.SGD(
            [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=TestLrScheduler.base_lr
        )

        def step_lr_step(base_lr, current_step, step_size, gamma):
            return base_lr * (gamma ** (current_step // step_size))

        gamma = 0.1
        step_size = 5
        step_lr = flow.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        for i in range(1, 21):
            step_lr.step()
            new_lr = step_lr_step(TestLrScheduler.base_lr, i, step_size, gamma)
            test_case.assertAlmostEqual(step_lr.get_last_lr()[0], new_lr, places=5)

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
                for base_lr, lmbda in zip(base_lrs, lambdas)
            ]

        lambda_lr = flow.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

        for i in range(1, 21):
            lambda_lr.step()
            new_lrs = lambda_lr_step(lambda_lr.base_lrs, i)
            for lr1, lr2 in zip(lambda_lr.get_last_lr(), new_lrs):
                test_case.assertAlmostEqual(lr1, lr2, places=5)


if __name__ == "__main__":
    unittest.main()
