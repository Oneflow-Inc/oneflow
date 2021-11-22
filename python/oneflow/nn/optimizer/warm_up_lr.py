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

from .optimizer import Optimizer
from .lr_scheduler import WarmUpLrScheduler


class WarmUpLR(WarmUpLrScheduler):
    def __init__(
        self,
        lrsch_or_optimizer,
        warmup_factor: float = 1.0 / 3,
        warmup_iters: int = 5,
        warmup_method="linear",
        last_step=-1,
        verbose=False,
    ):
        r"""Increasing the learning rate with a small warmup factor until the number of epoch 
        reaches the warmup_iters. You can assign an optimizer or a learning rate scheduler. 
        Notice that the warmup can happen simultaneously with learning rate scheduler. 

        When last_step = -1, it will set initial lr as lr. 

        Args:
            lrsch_or_optimizer ([type]): Learning rate scheduler or Optimizer
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

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, but "
                "got {}".format(warmup_method)
            )
        assert (
            warmup_iters > 0
        ), f"warmup_iters must greater than zero, but got {warmup_iters}"
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.warmup_method = warmup_method
        super().__init__(lrsch_or_optimizer, last_step, verbose)

    def get_lr(self):
        if self.warmup_method == "linear":
            if self.last_step < self.warmup_iters:
                multiplier = self.warmup_factor + (1.0 - self.warmup_factor) * (
                    self.last_step * 1.0 / self.warmup_iters
                )
                return [base_lr * multiplier for base_lr in self.base_lrs]
            else:
                return [base_lr for base_lr in self.base_lrs]
        elif self.warmup_method == "constant":
            if self.last_step < self.warmup_iters:
                return [base_lr * self.warmup_factor for base_lr in self.base_lrs]
            else:
                return [base_lr for base_lr in self.base_lrs]
        else:
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, but "
                "got {}".format(self.warmup_method)
            )

    def _generate_conf_for_graph(self, opt_confs):
        if self._inner_lr_sch is not None:
            self._inner_lr_sch._generate_conf_for_graph(opt_confs)
        if self.warmup_method == "linear":
            for opt_conf in opt_confs:
                warmup_conf = opt_conf.mutable_warmup_conf()
                warmup_conf.mutable_linear_conf().set_warmup_batches(self.warmup_iters)
                warmup_conf.mutable_linear_conf().set_start_multiplier(
                    self.warmup_factor
                )
        elif self.warmup_method == "constant":
            for opt_conf in opt_confs:
                warmup_conf = opt_conf.mutable_warmup_conf()
                warmup_conf.mutable_constant_conf().set_warmup_batches(
                    self.warmup_iters
                )
                warmup_conf.mutable_constant_conf().set_multiplier(self.warmup_factor)
        else:
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, but "
                "got {}".format(self.warmup_method)
            )
