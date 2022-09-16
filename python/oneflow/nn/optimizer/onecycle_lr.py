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

Reference from PyTorch implementation.
https://github.com/pytorch/pytorch/blob/0ec19db7ac88e307135100ddcfc418ae3925844f/torch/optim/lr_scheduler.py#L1430
"""
import math
from .optimizer import Optimizer
from .lr_scheduler import LRScheduler


class OneCycleLR(LRScheduler):
    """
    The interface is consistent with PyTorch.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.optim.lr_scheduler.OneCycleLR.html.

    Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    The default behaviour of this scheduler follows the fastai implementation of 1cycle, which
    claims that "unpublished work has shown even better results by using only two phases". To
    mimic the behaviour of the original paper instead, set ``three_phase=True``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        three_phase (bool): If ``True``, use a third phase of the schedule to annihilate the
            learning rate according to 'final_div_factor' instead of modifying the second
            phase (the first two phases will be symmetrical about the step indicated by
            'pct_start').
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. note::
        In Graph mode, param 'max_lr' doesn't support list input, and cycle_momentum=True is not supported.

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
            self.total_steps = total_steps
        else:
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError(f"Expected positive integer epochs, but got {epochs}")
            if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
                raise ValueError(
                    f"Expected positive integer steps_per_epoch, but got {steps_per_epoch}"
                )
            self.total_steps = epochs * steps_per_epoch

        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(
                f"Expected float between 0 and 1 pct_start, but got {pct_start}"
            )
        self.pct_start = pct_start

        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                f"anneal_strategy must by one of 'cos' or 'linear', instead got {anneal_strategy}"
            )
        elif anneal_strategy == "cos":
            self.anneal_func = lambda start, end, pct: end + (start - end) / 2.0 * (
                math.cos(math.pi * pct) + 1
            )
        elif anneal_strategy == "linear":
            self.anneal_func = lambda start, end, pct: (end - start) * pct + start
        self.anneal_strategy = anneal_strategy

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
        self.three_phase = three_phase

        max_lrs = self._format_param("max_lr", self.optimizer, max_lr)
        self.max_lrs = max_lrs
        if last_step == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group["initial_lr"] = max_lrs[idx] / div_factor
                group["max_lr"] = max_lrs[idx]
                group["min_lr"] = group["initial_lr"] / final_div_factor
        else:
            for idx, group in enumerate(self.optimizer.param_groups):
                if "max_lr" not in group or "min_lr" not in group:
                    raise KeyError(
                        "OneCycleLR expects optimizer has 'max_lr' and 'min_lr'."
                    )

        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if (
                "momentum" not in self.optimizer._default_options
                and "betas" not in self.optimizer._default_options
            ):
                raise ValueError(
                    "optimizer must support momentum with `cycle_momentum` option enabled"
                )
            self.use_beta1 = "betas" in self.optimizer._default_options
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

    def get_lr(self, _, step):
        lrs = []
        for group in self.optimizer.param_groups:
            for i, phase in enumerate(self._schedule_phases):
                start_step, end_step = phase["start_step"], phase["end_step"]
                if step <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step - start_step) / (end_step - start_step)
                    lr = self.anneal_func(
                        group[phase["start_lr"]], group[phase["end_lr"]], pct
                    )
                    if self.cycle_momentum:
                        computed_momentum = self.anneal_func(
                            group[phase["start_momentum"]],
                            group[phase["end_momentum"]],
                            pct,
                        )
                    break

            lrs.append(lr)
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
        if self.cycle_momentum == True:
            raise RuntimeError("cycle_momentum=True is not supported in graph mode.")
        if len(self.max_lrs) > 1:
            raise RuntimeError("Graph mode doesn't support list of max_lrs")

        anneal_strategy_dict = {
            "cos": 0,
            "linear": 1,
        }

        lr_conf.onecycle_lr_conf.SetInParent()
        onecycle_lr_conf = lr_conf.onecycle_lr_conf
        onecycle_lr_conf.max_lr = self.max_lrs[0]
        onecycle_lr_conf.div_factor = self.div_factor
        onecycle_lr_conf.final_div_factor = self.final_div_factor
        onecycle_lr_conf.total_steps = self.total_steps
        onecycle_lr_conf.three_phase = self.three_phase
        onecycle_lr_conf.pct_start = self.pct_start
        onecycle_lr_conf.anneal_strategy = anneal_strategy_dict[self.anneal_strategy]
