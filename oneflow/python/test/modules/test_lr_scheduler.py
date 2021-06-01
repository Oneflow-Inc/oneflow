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
    optimizer = flow.optim.SGD(
        [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=base_lr
    )

    def test_cosine_scheduler(test_case):
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
            TestLrScheduler.optimizer, steps=10, alpha=0.5
        )

        for i in range(1, 21):
            cosine_scheduler.step()
            new_lr = cosine_scheduler_step(TestLrScheduler.base_lr, i, steps, alpha)
            test_case.assertAlmostEqual(
                cosine_scheduler.get_last_lr()[0], new_lr, places=4
            )


if __name__ == "__main__":
    unittest.main()
