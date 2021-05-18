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
            for group in self._optimizer._param_groups:
                group.options["initial_lr"] = group.options["lr"]
        else:
            for i, group in enumerate(self._optimizer._param_groups):
                assert "initial_lr" in group.options, (
                    "param 'initial_lr' is not specified in "
                    f"param_groups[{i}] when resuming an optimizer"
                )

        self.base_lr = [
            group.options["initial_lr"] for group in self._optimizer._param_groups
        ]
        self.last_lr = list()
        self.last_step = last_step

        self.verbose = verbose
        self.step()

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "_optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def get_last_lr(self):
        return self.last_lr

    def print_lr(self, group_idx, lr):
        print(f"Adjusting learning rate of param_groups[{group_idx}] to {lr}")

    def step(self):
        self.last_step += 1
        self.last_lr = self.get_lr()

        for i, group in enumerate(self._optimizer._param_groups):
            group.options["lr"] = self.last_lr[i]
            if self.verbose:
                self.print_lr(i, self.last_lr[i])
