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


class LrScheduler(object):
    def __init__(self, optimizer, last_step=-1, verbose=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer object")
        self._optimizer = optimizer
        if last_step == -1:
            for group in self._optimizer.param_groups:
                group["initial_lr"] = group["lr"]
        else:
            for (i, group) in enumerate(self._optimizer.param_groups):
                assert (
                    "initial_lr" in group
                ), f"param 'initial_lr' is not specified in param_groups[{i}] when resuming an optimizer"
        self.base_lrs = [group["initial_lr"] for group in self._optimizer.param_groups]
        self.last_lr = list()
        self.last_step = last_step
        self.verbose = verbose
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for (key, value) in self.__dict__.items() if key != "_optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        """Compute learning rate using chainable form of the scheduler
        """
        raise NotImplementedError

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self.last_lr

    def print_lr(self, group_idx, lr):
        """Display the current learning rate.
        """
        print(f"Adjusting learning rate of param_groups[{group_idx}] to {lr}")

    def step(self):
        self.last_step += 1
        self.last_lr = self.get_lr()
        for (i, group) in enumerate(self._optimizer.param_groups):
            group["lr"] = self.last_lr[i]
            if self.verbose:
                self.print_lr(i, self.last_lr[i])
