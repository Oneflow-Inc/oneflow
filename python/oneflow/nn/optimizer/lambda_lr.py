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
import types

from .lr_scheduler import LRScheduler


class LambdaLR(LRScheduler):
    """
    Sets the learning rate of each parameter group to the initial lr times a given function.
    When last_step=-1, sets initial lr as lr.

    .. math::

        learning\\_rate = base\\_learning\\_rate*lambda(last\\_step)

    Args:
        optimizer(Optimizer): Wrapped optimizer.
        lr_lambda(function or list): A function which computes a multiplicative factor given an integer
            parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.
        last_step (int, optional): The index of last step. (default: -1)
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. (default: ``False``)

    For example:

    .. code-block:: python

        import oneflow as flow

        ...
        lambda1 = lambda step: step // 30
        lambda2 = lambda step: 0.95 * step
        lambda_lr = flow.optim.lr_scheduler.LambdaLR(optimizer, [lambda1, lambda2])
        for epoch in range(num_epoch):
            train(...)
            lambda_lr.step()

    """

    def __init__(self, optimizer, lr_lambda, last_step=-1, verbose=False):
        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            assert len(lr_lambda) == len(
                optimizer.param_groups
            ), f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
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
            for (key, value) in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)
        for (idx, fn) in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
        state_dict["lr_lambdas"] = lr_lambdas
        for (idx, fn) in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def step(self):
        """Performs a single learning rate schedule step.

        """
        self.last_step += 1
        lrs = []
        for (lmbda, base_lr) in zip(self.lr_lambdas, self.base_lrs):
            lrs.append(base_lr * lmbda(self.last_step))
        self.update_lrs(lrs)
