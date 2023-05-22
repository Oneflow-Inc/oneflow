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


class LRScheduler(object):
    def __init__(
        self, optimizer: Optimizer, last_step: int = -1, verbose: bool = False
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer object")

        self.optimizer = optimizer
        self.last_step = last_step
        self.verbose = verbose
        self._init_base_lrs()
        self.step()

    def state_dict(self):
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for (key, value) in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Load the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self, base_lr, step):
        """Compute learning rate using chainable form of the scheduler"""
        raise NotImplementedError

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    def print_lr(self, group, lr):
        """Display the current learning rate."""
        print(
            f"Last step {self.last_step} of {type(self)} adjusting learning rate "
            f"of param_groups[{group}] to {lr:.5f}"
        )

    def step(self):
        self.last_step += 1
        lrs = [self.get_lr(base_lr, self.last_step) for base_lr in self.base_lrs]
        self.update_lrs(lrs)

    def update_lrs(self, lrs):
        self._last_lr = []
        for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, lrs)):
            group["lr"] = lr
            self._last_lr.append(lr)
            if self.verbose:
                self.print_lr(i, lr)

    def _init_base_lrs(self):
        if self.last_step == -1:
            for group in self.optimizer.param_groups:
                if "initial_lr" not in group:
                    group.setdefault("initial_lr", group["lr"])
        else:
            for (i, group) in enumerate(self.optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        f"in param_groups[{i}] when resuming an optimizer"
                    )

        self.base_lrs = [group["initial_lr"] for group in self.optimizer.param_groups]
