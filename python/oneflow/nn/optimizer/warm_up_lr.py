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

    def generate_conf_for_graph(self, opt_confs):
        if self._inner_lr_sch is not None:
            self._inner_lr_sch.generate_conf_for_graph(opt_confs)
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
