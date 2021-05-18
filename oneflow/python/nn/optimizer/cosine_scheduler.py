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

from oneflow.python.oneflow_export import experimental_api, oneflow_export
from .lr_scheduler import LrScheduler


@oneflow_export("optim.lr_scheduler.CosineScheduler")
@experimental_api
class CosineScheduler(LrScheduler):
    r"""This operator creates a Cosine decayed learning rate scheduler.

    Before the steps are specified by user, the learning rate will be updated as:

    .. math::

        & cos\_decay = 0.5*(1+cos(\pi*\frac{current\_step}{steps}))

        & decay\_factor = (1-\alpha)*cos\_decay+\alpha

        & learning\_rate = base\_learning\_rate*decay\_factor

    After the steps specified by user, the learning rate will be :

    .. math::

        learning\_rate = {base\_learning\_rate}*{\alpha}

    Args:
        steps (int): The decay steps in the scheduler
        alpha (float, optional): The learning rate scale factor (:math:`\alpha`). Defaults to 0.0.

    For example:

    .. code-block:: python

        import oneflow.experimental as flow

        ...
        cosine_scheduler = flow.optim.lr_scheduler.CosineScheduler(optimizer, steps=100, alpha=0.0)
        for epoch in range(num_epoch):
            train(...)
            cosine_scheduler.step()

    """

    def __init__(
        self, optimizer, steps: int, alpha: float = 0.0, last_step=-1, verbose=False
    ):
        assert steps > 0, f"steps must greater than zero, but got {steps}"

        self.steps = steps
        self.alpha = alpha
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self):
        if self.last_step < self.steps:
            cos_decay = 0.5 * (1 + math.cos(math.pi * self.last_step / self.steps))
            decay_factor = (1 - self.alpha) * cos_decay + self.alpha
            return [base_lr * decay_factor for base_lr in self.base_lr]
        else:
            return [base_lr * self.alpha for base_lr in self.base_lr]
