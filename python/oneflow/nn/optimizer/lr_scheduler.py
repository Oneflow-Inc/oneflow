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


class LRScheduler(object):
    def __init__(
        self, optimizer: Optimizer, last_step: int = -1, verbose: bool = False
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer object")

        self._optimizer = optimizer
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
            key: value for (key, value) in self.__dict__.items() if key != "_optimizer"
        }

    def load_state_dict(self, state_dict):
        """Load the schedulers state.

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
        """Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def print_lr(self, group, lr):
        """Display the current learning rate.
        """
        print(
            f"Last step {self.last_step} of {type(self)} adjusting learning rate "
            f"of param_groups[{group}] to {lr:.5f}"
        )

    def step(self):
        self.last_step += 1
        lrs = self.get_lr()

        self._last_lr = []
        for i, (group, lr) in enumerate(zip(self._optimizer.param_groups, lrs)):
            group["lr"] = lr
            self._last_lr.append(lr)
            if self.verbose:
                self.print_lr(i, lr)

    def _init_base_lrs(self):
        if self.last_step == -1:
            for group in self._optimizer.param_groups:
                if "initial_lr" not in group:
                    group.setdefault("initial_lr", group["lr"])
        else:
            for (i, group) in enumerate(self._optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        f"in param_groups[{i}] when resuming an optimizer"
                    )

        self.base_lrs = [group["initial_lr"] for group in self._optimizer.param_groups]


class WarmUpLrScheduler(LRScheduler):
    def __init__(
        self, lrsch_or_optimizer, last_step=-1, verbose=False,
    ):
        self._inner_lr_sch = None
        if not isinstance(lrsch_or_optimizer, (LRScheduler, Optimizer)):
            raise TypeError(
                f"{type(lrsch_or_optimizer).__name__} is not an Optimizer object or a LRScheduler object."
            )
        if isinstance(lrsch_or_optimizer, LRScheduler):
            self._inner_lr_sch = lrsch_or_optimizer
            # the _inner_lr_sch has called step() in it's __init__
            if self._inner_lr_sch.last_step != last_step + 1:
                raise ValueError(
                    "The last_step of this warmup lr scheduler must match that of the wrapped lr scheduler"
                    f" ({last_step + 1} vs. {self._inner_lr_sch.last_step})"
                )

            for i, saved_lr in enumerate(self._inner_lr_sch._saved_lr):
                # WarmUpLrScheduler will restore lr changed by step() in self._inner_lr_sch.__init___()
                self._inner_lr_sch._optimizer.param_groups[i]["lr"] = saved_lr

            super().__init__(lrsch_or_optimizer._optimizer, last_step, verbose)
        else:
            super().__init__(lrsch_or_optimizer, last_step, verbose)

    def step(self):
        if self.last_step + 1 < self.warmup_iters or self._inner_lr_sch is None:
            # warmup lr_scheduler step
            super().step()
        else:
            # sync right last_step to inner lr_scheduler
            self._inner_lr_sch.last_step = self.last_step
            # inner lr_scheduler step
            self._inner_lr_sch.step()
            # get right last_step from inner lr_scheduler
            self.last_step = self._inner_lr_sch.last_step

    def state_dict(self):
        """Return the state of the scheduler as a :class:`dict`.
        """
        state = {
            key: value for (key, value) in self.__dict__.items() if key != "_optimizer"
        }
        if self._inner_lr_sch is not None:
            state["_inner_lr_sch"] = self._inner_lr_sch.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        if self._inner_lr_sch is not None:
            assert "_inner_lr_sch" in state_dict
            inner_lr_sch_state = state_dict.pop("_inner_lr_sch")
            self._inner_lr_sch.load_state_dict(inner_lr_sch_state)
        self.__dict__.update(state_dict)
        # Resume _inner_lr_sch because that we should not change `state_dict`
        if self._inner_lr_sch is not None:
            state_dict["_inner_lr_sch"] = inner_lr_sch_state


class _scheduler_with_step(object):
    def __init__(self, scheduler, step):
        self._scheduler = scheduler
        self._step = step
        self._origin_step = scheduler.last_step

    def __enter__(self):
        self._scheduler.last_step = self._step
        return self._scheduler

    def __exit__(self, type, value, traceback):
        self._scheduler.last_step = self._origin_step
