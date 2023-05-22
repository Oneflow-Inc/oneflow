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
from ...optim.optimizer import Optimizer
from .lr_scheduler import LRScheduler


class ExponentialLR(LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_step (int): The index of last step. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_step: int = -1,
        verbose: bool = False,
    ):
        assert isinstance(optimizer, Optimizer)
        if gamma <= 0.0:
            raise ValueError(f"'gamma' must be greater than zero, but got {gamma}")

        self.gamma = gamma
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self, base_lr, step):
        return base_lr * (self.gamma ** step)

    def _generate_conf_for_graph(self, lr_conf):
        lr_conf.step_conf.SetInParent()
        step_conf = lr_conf.step_conf
        step_conf.step_size = 1
        step_conf.gamma = self.gamma
