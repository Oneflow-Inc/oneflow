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

from .lr_scheduler import LRScheduler


class PolynomialLR(LRScheduler):
    r"""
    This operator creates a polynomial decayed learning rate scheduler.
    The learning rate will be updated as follows:

    If cycle is `True`, the equation is:

    .. math::
        \begin{aligned}
           & decay\_batch = decay\_batch*ceil(\frac{current\_batch}{decay\_batch}) \\
           & learning\_rate = (base\_lr-end\_lr)*(1-\frac{current\_batch}{decay\_batch})^{power}+end\_lr
        \end{aligned}

    If cycle is `False`, the equation is:

    .. math::
        \begin{aligned}
           & current\_batch = min(decay\_batch, current\_batch) \\
           & learning\_rate = (base\_lr-end\_lr)*(1-\frac{current\_batch}{decay\_batch})^{power}+end\_lr
        \end{aligned}

    Args:
        optimizer (Optimizer): Wrapper optimizer.
        decay_batch (int): The decayed steps.
        end_learning_rate (float, optional): The final learning rate. Defaults to 0.0001.
        power (float, optional): The power of polynomial. Defaults to 1.0.
        cycle (bool, optional): If cycle is True, the scheduler will decay the learning rate every decay steps. Defaults to False.

    For example:

    .. code-block:: python

        import oneflow as flow
       
        ... 
        polynomial_scheduler = flow.optim.lr_scheduler.PolynomialLR(
            optimizer, decay_batch=5, end_learning_rate=0.00001, power=2
            )

        for epoch in range(num_epoch):
            train(...)
            polynomial_scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        decay_batch: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        last_step: int = -1,
        verbose: bool = False,
    ):
        assert (
            decay_batch > 0
        ), f"decay_batch must greater than zero, but got {decay_batch}"
        self.max_decay_steps = decay_batch
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self, base_lr, step):
        decay_batch = self.max_decay_steps
        cur_batch = step
        if self.cycle:
            if cur_batch == 0:
                cur_batch = 1
            decay_batch = decay_batch * math.ceil(cur_batch / decay_batch)
        else:
            cur_batch = min(cur_batch, decay_batch)

        factor = (1 - cur_batch / decay_batch) ** (self.power)
        return (base_lr - self.end_learning_rate) * factor + self.end_learning_rate

    def _generate_conf_for_graph(self, lr_conf):
        lr_conf.polynomial_conf.SetInParent()
        polynomial_conf = lr_conf.polynomial_conf
        polynomial_conf.decay_batches = self.max_decay_steps
        polynomial_conf.end_learning_rate = self.end_learning_rate
        polynomial_conf.power = self.power
        polynomial_conf.cycle = self.cycle
