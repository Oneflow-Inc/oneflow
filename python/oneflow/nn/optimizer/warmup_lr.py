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
import numpy as np
from typing import Union
from .optimizer import Optimizer
from .lr_scheduler import LRScheduler, _scheduler_with_step
from .sequential_lr import SequentialLR
from .constant_lr import ConstantLR
from .linear_lr import LinearLR


class WarmupLR(SequentialLR):
    r"""Increasing the learning rate with a small warmup factor until the number of epoch 
    reaches the warmup_iters. You can assign an optimizer or a learning rate scheduler. 
    Notice that the warmup can happen simultaneously with learning rate scheduler. 

    Args:
        scheduler_or_optimizer ([type]): Wrapped learning rate scheduler or optimizer
        warmup_factor (float, optional): The warmup factor. Defaults to 1.0/3.
        warmup_iters (int, optional): The number of warmup steps. Defaults to 5.
        warmup_method (str, optional): The method of warmup, you can choose "linear" or "constant". 
            In linear mode, the multiplication factor starts with warmup_factor in the first epoch and then inreases linearly to reach 1. Defaults to "linear".
        last_step (int, optional): The index of the last step. Defaults to -1.
        verbose (bool, optional): If True, it prints a message to stdout for each update step. Defaults to False.

    Raises:
        ValueError: The warmup method should be one of the "constant" and "linear" 

    For example: 

    Example 1: 

    .. code:: python 

        # lr = 0.0005    if epoch == 0
        # lr = 0.0005    if epoch == 1
        # lr = 0.0005    if epoch == 2
        # lr = 0.0005    if epoch == 3
        # lr = 0.0005    if epoch == 4
        # lr = 0.001     if epoch >= 5
        of_sgd = flow.optim.SGD(parameters, lr=0.001)
        constant_warmup_lr = flow.optim.lr_scheduler.WarmUpLR(
            of_sgd, warmup_factor=0.5, warmup_iters=5, warmup_method="constant"
        )
        ...

    Example 2: 

    .. code:: python 

        # lr = 0.0005    if epoch == 0
        # lr = 0.0006    if epoch == 1
        # lr = 0.0007    if epoch == 2
        # lr = 0.0008    if epoch == 3
        # lr = 0.0009    if epoch == 4
        # lr = 0.001    if epoch >= 5
        of_sgd = flow.optim.SGD(parameters, lr=0.001)
        constant_warmup_lr = flow.optim.lr_scheduler.WarmUpLR(
            of_sgd, warmup_factor=0.5, warmup_iters=5, warmup_method="linear"
        )
        ...

    Example 2: 

    .. code:: python 

        # lr = 0.0005    if epoch == 0
        # lr = 0.00075   if epoch == 1
        # Above is WarmUpLR, then we start CosineDecayLR
        # lr = 0.000689  if epoch == 2
        # lr = 0.000410  if epoch == 3
        # ....
        of_sgd = flow.optim.SGD(parameters, lr=0.001)
        alpha = 0.1
        decay_steps = 5
        cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(
            of_sgd, decay_steps=decay_steps, alpha=alpha
        )
        linear_warmup_cosine_lr = flow.optim.lr_scheduler.WarmUpLR(
            cosine_decay_lr, warmup_factor=0.5, warmup_iters=2, warmup_method="linear"
        )
        ...
    """

    def __init__(
        self,
        scheduler_or_optimizer: Union[LRScheduler, Optimizer],
        warmup_factor: float = 1.0 / 3,
        warmup_iters: int = 5,
        warmup_method: str = "linear",
        warmup_prefix: bool = False,
        last_step=-1,
        verbose=False,
    ):
        if not isinstance(scheduler_or_optimizer, (LRScheduler, Optimizer)):
            raise ValueError(
                "'scheduler_or_optimizer' must be a LRScheduler or an Optimizer, but got "
                f"{type(scheduler_or_optimizer)}"
            )

        if warmup_method not in ("linear", "constant"):
            raise ValueError(
                f"'warmup_method' must be 'linear' or 'constant', but got {warmup_method}"
            )

        if isinstance(scheduler_or_optimizer, LRScheduler):
            opt = scheduler_or_optimizer._optimizer
            self.successor_scheduler = scheduler_or_optimizer
        else:
            opt = scheduler_or_optimizer
            self.successor_scheduler = None

        if self.successor_scheduler is None and warmup_iters == 0:
            raise ValueError(
                "When 'scheduler_or_optimizer' is an optimizer warmup_iters can't be equal to 0"
            )

        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.warmup_prefix = warmup_prefix
        # manually init optimizer, last_step, base_lrs first
        self._optimizer = opt
        self.last_step = last_step
        self.verbose = verbose
        self._init_base_lrs()
        self._init_warmup_scheduler()
        self._init_seq_scheduler()

    def _init_warmup_scheduler(self):
        self.warmup_scheduler = None

        if self.warmup_iters <= 0:
            return

        if self.warmup_method == "linear":
            if self.successor_scheduler and self.warmup_prefix is False:
                with _scheduler_with_step(
                    self.successor_scheduler, self.warmup_iters
                ) as scheduler:
                    lrs = scheduler.get_lr()

                end_factor = [lr / base_lr for lr, base_lr in zip(lrs, self.base_lrs)]
                assert len(end_factor) > 0
                assert np.isclose(end_factor, end_factor[0]).all()
                end_factor = end_factor[0]
            else:
                end_factor = 1.0

            self.warmup_scheduler = LinearLR(
                self._optimizer,
                start_factor=self.warmup_factor,
                end_factor=end_factor,
                total_iters=self.warmup_iters,
                last_step=self.last_step,
                verbose=self.verbose,
            )
        else:  # "constant"
            self.warmup_scheduler = ConstantLR(
                self._optimizer,
                factor=self.warmup_factor,
                total_iters=self.warmup_iters,
                last_step=self.last_step,
                verbose=self.verbose,
            )

    def _init_seq_scheduler(self):
        if self.warmup_scheduler and self.successor_scheduler:
            schedulers = [self.warmup_scheduler, self.successor_scheduler]
            milestones = [self.warmup_iters]
            interval_rescaling = [False, self.warmup_prefix]
        elif self.warmup_scheduler:
            schedulers = [self.warmup_scheduler]
            milestones = []
            interval_rescaling = False
        elif self.successor_scheduler:
            schedulers = [self.successor_scheduler]
            milestones = []
            interval_rescaling = False
        else:
            raise ValueError("No scheduler can work")

        super().__init__(
            self._optimizer,
            schedulers=schedulers,
            milestones=milestones,
            interval_rescaling=interval_rescaling,
            last_step=self.last_step,
            verbose=self.verbose,
        )

    def state_dict(self):
        # exclude optimizer and nested schedulers
        exclude_attrs = (
            "_optimizer",
            "schedulers",  # in parent SequentialLR
            "warmup_scheduler",
            "successor_scheduler",
        )

        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in exclude_attrs
        }

        state_dict["warmup_scheduler"] = self.warmup_scheduler.state_dict()
        state_dict["successor_scheduler"] = self.successor_scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        if "warmup_scheduler" in state_dict:
            warmup_scheduler_state = state_dict.pop("warmup_scheduler")
            if self.warmup_scheduler:
                self.warmup_scheduler.load_state_dict(warmup_scheduler_state)

        if "successor_scheduler" in state_dict:
            successor_scheduler_state = state_dict.pop("successor_scheduler")
            if self.successor_scheduler:
                self.successor_scheduler.load_state_dict(successor_scheduler_state)

        self.__dict__.update(state_dict)

    def _generate_conf_for_graph(self, opt_confs):
        if self.warmup_scheduler:
            for op_conf in opt_confs:
                warmup_conf = op_conf.mutable_warmup_conf()
                warmup_conf.set_warmup_batches(self.warmup_iters)
                warmup_conf.set_warmup_factor(self.warmup_factor)
                if self.warmup_method == "linear":
                    warmup_conf.mutable_linear_conf()
                else:
                    warmup_conf.mutable_constant_conf()
                warmup_conf.set_prefix(self.warmup_prefix)

        if self.successor_scheduler:
            self.successor_scheduler._generate_conf_for_graph(opt_confs)
