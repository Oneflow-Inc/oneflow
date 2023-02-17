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


class MultiplicativeLR(LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given
    in the specified function. When last_epoch=-1, sets initial lr as lr.

    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_step (int): The index of last step. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    For example:

    .. code-block:: python

        import oneflow as flow

        ...
        lmbda = lambda epoch: 0.95
        step_lr = flow.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        for epoch in range(num_epoch):
            train(...)
            step_lr.step()
    """

    def __init__(self, optimizer, lr_lambda, last_step=-1, verbose=False):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected {} lr_lambdas, but got {}".format(
                        len(optimizer.param_groups), len(lr_lambda)
                    )
                )
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_step, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
        state_dict["lr_lambdas"] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def step(self):
        """Performs a single learning rate schedule step.

        """
        self.last_step += 1
        if self.last_step > 0:
            lrs = [
                group["lr"] * lmbda(self.last_step)
                for lmbda, group in zip(self.lr_lambdas, self.optimizer.param_groups)
            ]
        else:
            lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.update_lrs(lrs)
