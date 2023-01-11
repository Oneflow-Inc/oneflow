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
import bisect

from ...optim.optimizer import Optimizer
from .lr_scheduler import LRScheduler


class MultiStepLR(LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma once the number of step
    reaches one of the milestones. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler.When last_step=-1, sets initial lr as lr.

    Args:
        optimizer(Optimizer): Wrapped optimizer.
        milestones(list): List of step indices. Must be increasing
        gamma (float, optional): Multiplicative factor of learning rate decay. (default: 0.1)
        last_step (int, optional): The index of last step. (default: -1)
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. (default: ``False``)

    For example:

    .. code-block:: python

        import oneflow as flow

        ...
        multistep_lr = flow.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        for epoch in range(num_epoch):
            train(...)
            multistep_lr.step()

    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list,
        gamma: float = 0.1,
        last_step: int = -1,
        verbose: bool = False,
    ):
        for i in range(1, len(milestones)):
            assert (
                milestones[i] > milestones[i - 1]
            ), f"values in `list` milestone must be increasing, but got {milestones}"
        assert gamma > 0.0, f"gamma must greater than zero, but got {gamma}"
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self, base_lr, step):
        sect = bisect.bisect_right(self.milestones, step)
        factor = self.gamma ** sect
        return base_lr * factor

    def _generate_conf_for_graph(self, lr_conf):
        lr_conf.multi_step_conf.SetInParent()
        multi_step_conf = lr_conf.multi_step_conf
        for milestone in self.milestones:
            multi_step_conf.milestones.append(milestone)
        multi_step_conf.gamma = self.gamma
