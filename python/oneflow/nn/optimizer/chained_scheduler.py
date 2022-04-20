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
from .lr_scheduler import LRScheduler


class ChainedScheduler(LRScheduler):
    """Chains list of learning rate schedulers. It takes a list of chainable learning
    rate schedulers and performs consecutive step() functions belong to them by just
    one call.

    Args:
        schedulers (list): List of chained schedulers.

    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.09     if step == 0
        >>> # lr = 0.081    if step == 1
        >>> # lr = 0.729    if step == 2
        >>> # lr = 0.6561   if step == 3
        >>> # lr = 0.59049  if step >= 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = ChainedScheduler([scheduler1, scheduler2])
        >>> for _ in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, schedulers):
        if not isinstance(schedulers, (list, tuple)) or any(
            not isinstance(s, LRScheduler) for s in schedulers
        ):
            raise ValueError("ChainedScheduler expects a list of schedulers")

        if len(schedulers) == 0:
            raise ValueError("length of list of schedulers must be greater than 0")

        opt = schedulers[0].optimizer

        for i in range(1, len(schedulers)):
            if schedulers[i].optimizer != opt:
                raise ValueError(
                    "ChainedScheduler expects all schedulers to belong to the same optimizer, but "
                    f"got schedulers at index {0} and {i} to be different"
                )

        self.schedulers = list(schedulers)
        super().__init__(optimizer=opt)

    def step(self):
        self.last_step += 1
        lrs = self.schedulers[0].base_lrs.copy()
        for scheduler in self.schedulers:
            for i, lr in enumerate(lrs):
                lrs[i] = scheduler.get_lr(lr, self.last_step)

            scheduler.last_step = self.last_step

        self.update_lrs(lrs)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
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
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        scheduler_states = state_dict.pop("schedulers")
        self.__dict__.update(state_dict)
        # avoid side effect of calling load_state_dict twice
        state_dict["schedulers"] = scheduler_states

        for i, s in enumerate(scheduler_states):
            self.schedulers[i].load_state_dict(s)

    def _generate_conf_for_graph(self, lr_conf):
        raise NotImplementedError("ChainedScheduler is not supported in graph mode yet")
