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
from .lr_scheduler import LrScheduler


class ExponentialLR(LrScheduler):
    """
    Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma: float, last_step=-1, verbose=False):
        assert gamma > 0.0, f"gamma must greater than zero, but got {gamma}"
        self.gamma = gamma
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self):
        if self.last_step == 0:
            return self.base_lrs
        return [group["lr"] * self.gamma for group in self._optimizer.param_groups]

    def _generate_conf_for_graph(self, opt_confs):
        for opt_conf in opt_confs:
            learning_rate_decay_conf = opt_conf.mutable_learning_rate_decay()
            learning_rate_decay_conf.mutable_step_conf().set_gamma(self.gamma)
