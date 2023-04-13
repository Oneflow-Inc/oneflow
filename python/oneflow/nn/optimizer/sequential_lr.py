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
from typing import Sequence, Union
from ...optim.optimizer import Optimizer
from .lr_scheduler import LRScheduler


class SequentialLR(LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given step.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        interval_rescaling (bool or list): Each scheduler has a corresponding 'interval_rescaling'.
            If it is set to True, scheduler will start and end at the same values as it would
            if it were the only scheduler, otherwise all schedulers share the same step.
            Default is False for all schedulers.
        last_step (int): The index of last step. Default: -1.
        verbose (bool): Default: False. Print lr if is set to True.

    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if step == 0
        >>> # lr = 0.1     if step == 1
        >>> # lr = 0.9     if step == 2
        >>> # lr = 0.81    if step == 3
        >>> # lr = 0.729   if step == 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = SequentialLR(self.opt, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for step in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: Sequence[LRScheduler],
        milestones: Sequence[int],
        interval_rescaling: Union[Sequence[bool], bool] = False,
        last_step: int = -1,
        verbose: bool = False,
    ):
        assert isinstance(optimizer, Optimizer)
        assert isinstance(schedulers, (list, tuple))
        assert isinstance(milestones, (list, tuple))

        if len(schedulers) == 0:
            raise ValueError("Sequential Schedulers expects at least one scheduler")

        for i in range(len(schedulers)):
            if schedulers[i].optimizer != optimizer:
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    f"got schedulers at index {i} to be different than the optimizer passed in."
                )

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                f"Sequential Schedulers expects number of schedulers provided to be one more "
                f"than the number of milestone points, but got number of schedulers {len(schedulers)} "
                f"and the number of milestones to be equal to {len(milestones)}"
            )

        if isinstance(interval_rescaling, (list, tuple)):
            if len(interval_rescaling) != len(milestones):
                raise ValueError(
                    "'interval_rescaling' expects a bool or a list of bool with length be equal to "
                    f"the number of milestones, but got number of milestones {len(milestones)} "
                    f"and the length of list of interval_rescaling {len(interval_rescaling)}"
                )

            assert all([isinstance(r, bool) for r in interval_rescaling])
        else:
            assert isinstance(interval_rescaling, bool)
            interval_rescaling = [interval_rescaling] * (len(milestones))

        self.schedulers = list(schedulers)
        self.milestones = list(milestones)
        self.interval_rescaling = list(interval_rescaling)
        super().__init__(optimizer, last_step, verbose)

    def step(self):
        self.last_step += 1
        cur_step = self.last_step
        s_i = bisect.bisect_right(self.milestones, cur_step)
        if s_i > 0 and self.interval_rescaling[s_i - 1]:
            cur_step = self.last_step - self.milestones[s_i - 1]

        scheduler = self.schedulers[s_i]
        scheduler.last_step = cur_step
        lrs = [scheduler.get_lr(base_lr, cur_step) for base_lr in self.base_lrs]
        self.update_lrs(lrs)

    def state_dict(self):
        # exclude optimizer and nested schedulers
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "schedulers")
        }
        state_dict["schedulers"] = [None] * len(self.schedulers)
        for i, s in enumerate(self.schedulers):
            state_dict["schedulers"][i] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        scheduler_states = state_dict.pop("schedulers")
        self.__dict__.update(state_dict)
        # avoid side effect of calling load_state_dict twice
        state_dict["schedulers"] = scheduler_states

        for i, s in enumerate(scheduler_states):
            self.schedulers[i].load_state_dict(s)

    def _generate_conf_for_graph(self, lr_conf):
        lr_conf.sequential_scheduler_conf.SetInParent()
        seq_lr_conf = lr_conf.sequential_scheduler_conf

        for scheduler in self.schedulers:
            scheduler._generate_conf_for_graph(seq_lr_conf.schedulers.add())

        for m in self.milestones:
            seq_lr_conf.milestones.append(m)

        for r in self.interval_rescaling:
            seq_lr_conf.interval_rescaling.append(r)
