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
from .lr_scheduler import WarmupLrScheduler


class ConstantWarmupLR(WarmupLrScheduler):
    def __init__(
        self,
        lrsch_or_optimizer,
        steps: int,
        multiplier: float = 0.0,
        last_step=-1,
        verbose=False,
    ):
        assert steps > 0, f"steps must greater than zero, but got {steps}"
        self.steps = steps
        self.multiplier = multiplier
        super().__init__(lrsch_or_optimizer, last_step, verbose)

    def get_lr(self):
        if self.last_step < self.steps:
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]

    def generate_conf_for_graph(self, opt_confs):
        if self._lr_sch is not None:
            self._lr_sch.generate_conf_for_graph(opt_confs)
        for opt_conf in opt_confs:
            warmup_conf = opt_conf.mutable_warmup_conf()
            warmup_conf.mutable_constant_conf().set_warmup_batches(self.steps)
            warmup_conf.mutable_constant_conf().set_multiplier(self.multiplier)
