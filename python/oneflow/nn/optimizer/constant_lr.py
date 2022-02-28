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
from .optimizer import Optimizer
from .lr_scheduler import LRScheduler


class ConstantLR(LRScheduler):
    """Decays the learning rate of each parameter group by a small constant factor until the
    number of step reaches a pre-defined milestone: total_iters.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler decays the learning rate.
            Default: 5.
        last_step (int): The last step. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each step. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if step == 0
        >>> # lr = 0.025   if step == 1
        >>> # lr = 0.025   if step == 2
        >>> # lr = 0.025   if step == 3
        >>> # lr = 0.05    if step >= 4
        >>> scheduler = ConstantLR(self.opt, factor=0.5, total_iters=4)
        >>> for step in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_step: int = -1,
        verbose: bool = False,
    ):
        assert isinstance(optimizer, Optimizer)

        if factor > 1.0 or factor < 0:
            raise ValueError(
                "Constant multiplicative factor expected to be between 0 and 1."
            )

        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self):
        if self.last_step < self.total_iters:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]

    def _generate_conf_for_graph(self, opt_confs):
        for opt_conf in opt_confs:
            learning_rate_decay_conf = opt_conf.mutable_learning_rate_decay()
            constant_lr_conf = learning_rate_decay_conf.mutable_constant_lr_conf()
            constant_lr_conf.set_factor(self.factor)
            constant_lr_conf.set_total_iters(self.total_iters)
