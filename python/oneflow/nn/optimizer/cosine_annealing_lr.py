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

from .lr_scheduler import LrScheduler


class CosineAnnealingLR(LrScheduler):
    r"""
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html?highlight=cosine#torch.optim.lr_scheduler.CosineAnnealingLR

    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_step=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_step (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self, optimizer, T_max: int, eta_min: float = 0.0, last_step=-1, verbose=False
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self):
        if self.last_step == 0:
            return [group["lr"] for group in self._optimizer.param_groups]
        elif (self.last_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self._optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_step / self.T_max))
            / (1 + math.cos(math.pi * (self.last_step - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self._optimizer.param_groups
        ]

    def _generate_conf_for_graph(self, opt_confs):
        for opt_conf in opt_confs:
            learning_rate_decay_conf = opt_conf.mutable_learning_rate_decay()
            learning_rate_decay_conf.mutable_cosine_annealing_conf().set_t_max(
                self.T_max
            )
            learning_rate_decay_conf.mutable_cosine_annealing_conf().set_eta_min(
                self.eta_min
            )
