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
from typing import Callable
from .optimizer import Optimizer
from .lr_scheduler import LRScheduler


class OneCycleLR(LRScheduler):
    """
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr,
        total_steps=None,
        epochs=None,
        steps_per_epoch=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_step: int = -1,
        verbose: bool = False,
    ):
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self.max_lr = max_lr
        if total_steps is None and (epochs is None or steps_per_epoch is None):
            raise ValueError(
                "You must define either total_steps OR (epochs AND steps_per_epoch)"
            )
        elif total_steps is not None:
            if not isinstance(total_steps, int) or total_steps <= 0:
                raise ValueError(
                    "Expected positive integer total_steps, but got {}".format(
                        total_steps
                    )
                )
            self.total_step = total_steps
        else:
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError(f"Expected positive integer epochs, but got {epochs}")
            if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
                raise ValueError(
                    f"Expected positive integer steps_per_epoch, but got {steps_per_epoch}"
                )
            self.total_step = epochs * steps_per_epoch

        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(
                f"Expected float between 0 and 1 pct_start, but got {pct_start}"
            )

        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                f"anneal_strategy must by one of 'cos' or 'linear', instead got {anneal_strategy}"
            )
        elif anneal_strategy == "cos":
            self.anneal_func = lambda start, end, pct: end + (start - end) / (
                2.0 * math.cos(math.pi * pct) + 1
            )
        elif anneal_strategy == "linear":
            self.anneal_func = lambda start, end, pct: (end - start) * pct + start

        if three_phase:
            self._schedule_phases = [
                {
                    "start_step": 0,
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "start_step": float(pct_start * self.total_steps) - 1,
                    "end_step": float(2 * pct_start * self.total_steps) - 2,
                    "start_lr": "max_lr",
                    "end_lr": "initial_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
                {
                    "start_step": float(2 * pct_start * self.total_steps) - 2,
                    "end_step": self.total_steps - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "max_momentum",
                },
            ]
        else:
            self._schedule_phases = [
                {
                    "start_step": 0,
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "start_step": float(pct_start * self.total_steps) - 1,
                    "end_step": self.total_steps - 1,
                    "start_lr": "max_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
            ]

        max_lrs = self._format_param("max_lr", self.optimizer, max_lr)
        if last_step == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group["initial_lr"] = max_lrs[idx] / div_factor
                group["max_lr"] = max_lrs[idx]
                group["min_lr"] = group["initial_lr"] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if (
                "momentum" not in self.optimizer.defaults
                and "betas" not in self.optimizer.defaults
            ):
                raise ValueError(
                    "optimizer must support momentum with `cycle_momentum` option enabled"
                )
            self.use_beta1 = "betas" in self.optimizer.defaults
            max_momentums = self._format_param("max_momentum", optimizer, max_momentum)
            base_momentums = self._format_param(
                "base_momentum", optimizer, base_momentum
            )
            if last_step == -1:
                for m_momentum, b_momentum, group in zip(
                    max_momentums, base_momentums, optimizer.param_groups
                ):
                    if self.use_beta1:
                        _, beta2 = group["betas"]
                        group["betas"] = (m_momentum, beta2)
                    else:
                        group["momentum"] = m_momentum
                    group["max_momentum"] = m_momentum
                    group["base_momentum"] = b_momentum

        super().__init__(optimizer, last_step, verbose)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} values for {}, got {}".format(
                        len(optimizer.param_groups), name, len(param)
                    )
                )
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self, base_lr, step):
        lrs = []
        for group in self.optimizer.param_groups:
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                start_step, end_step = phase["start_step"], phase["end_step"]
                if step <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step - start_step) / (end_step - start_step)
                    computed_lr = self.anneal_func(
                        group[phase["start_lr"]], group[phase["end_lr"]], pct
                    )
                    if self.cycle_momentum:
                        computed_momentum = self.anneal_func(
                            group[phase["start_momentum"]],
                            group[phase["end_momentum"]],
                            pct,
                        )
                    break

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group["betas"]
                    group["betas"] = (computed_momentum, beta2)
                else:
                    group["momentum"] = computed_momentum

        return lrs

    def step(self):
        self.last_step += 1
        lrs = self.get_lr(None, self.last_step)
        self.update_lrs(lrs)

    def _generate_conf_for_graph(self, lr_conf):
        pass
