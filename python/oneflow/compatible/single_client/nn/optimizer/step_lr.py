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


class StepLR(LrScheduler):
    """
    Decays the learning rate of each parameter group by gamma every step_size steps.
    Notice that such decay can happen simultaneously with other changes to the learning
    rate fromoutside this scheduler. When last_step=-1, sets initial lr as lr.

    Args:
        optimizer(Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float, optional): Multiplicative factor of learning rate decay. (default: 0.1)
        last_step (int, optional): The index of last step. (default: -1)
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. (default: ``False``)

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client.experimental as flow

        ...
        step_lr = flow.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        for epoch in range(num_epoch):
            train(...)
            step_lr.step()

    """

    def __init__(
        self, optimizer, step_size: int, gamma: float = 0.1, last_step=-1, verbose=False
    ):
        assert step_size > 0, f"step_size must greater than zero, but got {step_size}"
        assert gamma > 0.0, f"gamma must greater than zero, but got {gamma}"
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self):
        if self.last_step == 0 or self.last_step % self.step_size != 0:
            return [group["lr"] for group in self._optimizer.param_groups]
        else:
            return [group["lr"] * self.gamma for group in self._optimizer.param_groups]
