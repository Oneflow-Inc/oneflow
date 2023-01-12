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
from ...optim.optimizer import Optimizer
from .lr_scheduler import LRScheduler


class CosineAnnealingWarmRestarts(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of steps since the last restart and :math:`T_{i}` is the number
    of steps between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        decay_rate (float, optional): Decay rate every restarts.
        restart_limit (int, optional): The limit of restarts. 0 indicate unlimited restarts. Default: 0.
        last_step (int, optional): The index of last step. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        decay_rate: float = 1.0,
        restart_limit: int = 0,
        last_step: int = -1,
        verbose: bool = False,
    ):
        assert isinstance(optimizer, Optimizer)
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")

        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_rate = decay_rate
        self.restart_limit = restart_limit

        super().__init__(optimizer, last_step, verbose)

    def get_lr(self, base_lr, step):
        if self.T_mult > 1:
            epoch = math.floor(
                math.log(1 - step / self.T_0 * (1 - self.T_mult), self.T_mult)
            )
            epoch_steps = self.T_mult ** epoch * self.T_0
            step_in_epoch = (
                step - (1 - self.T_mult ** epoch) / (1 - self.T_mult) * self.T_0
            )
        else:
            epoch = step // self.T_0
            epoch_steps = self.T_0
            step_in_epoch = step - (epoch_steps * epoch)

        gamma = self.decay_rate ** epoch
        if self.restart_limit == 0 or (
            self.restart_limit > 0 and epoch < self.restart_limit
        ):
            cos_decay = 0.5 * (1 + math.cos(math.pi * step_in_epoch / epoch_steps))
            return self.eta_min + (base_lr * gamma - self.eta_min) * cos_decay

        return self.eta_min

    def _generate_conf_for_graph(self, lr_conf):
        lr_conf.cosine_annealing_warm_restarts_conf.SetInParent()
        cosa_warm_restarts_conf = lr_conf.cosine_annealing_warm_restarts_conf
        cosa_warm_restarts_conf.t_initial = self.T_0
        cosa_warm_restarts_conf.t_mult = self.T_mult
        cosa_warm_restarts_conf.eta_min = self.eta_min
        cosa_warm_restarts_conf.decay_rate = self.decay_rate
        cosa_warm_restarts_conf.restart_limit = self.restart_limit
