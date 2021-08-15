import math
import unittest

import oneflow as flow
import oneflow.unittest
from oneflow.nn.parameter import Parameter


@flow.unittest.skip_unless_1n1d()
class TestGraphLrScheduler(flow.unittest.TestCase):
    base_lr = 1.0

    def test_cosine_annealing_lr(test_case):
        optimizer = flow.optim.SGD(
            [{"params": [Parameter(flow.Tensor([1.0]))]}], lr=TestLrScheduler.base_lr
        )

        def cosine_annealing_lr_step(base_lr, current_step, steps, alpha):
            if current_step < steps:
                cos_decay = 0.5 * (1 + math.cos(math.pi * current_step / steps))
                decay_factor = (1 - alpha) * cos_decay + alpha
                return base_lr * decay_factor
            else:
                return base_lr * alpha

        alpha = 0.5
        steps = 10
        cosine_annealing_lr = flow.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, steps=steps, alpha=alpha
        )
        for i in range(1, 21):
            cosine_annealing_lr.step()
            new_lr = cosine_annealing_lr_step(TestLrScheduler.base_lr, i, steps, alpha)
            test_case.assertAlmostEqual(
                cosine_annealing_lr.get_last_lr()[0], new_lr, places=4
            )


if __name__ == "__main__":
    unittest.main()