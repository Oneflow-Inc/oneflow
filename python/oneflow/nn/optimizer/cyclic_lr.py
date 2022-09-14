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
from typing import Callable
from .optimizer import Optimizer
from .lr_scheduler import LRScheduler


class CyclicLR(LRScheduler):
    r"""
    The interface is consistent with PyTorch.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.optim.lr_scheduler.CyclicLR.html.

    Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by :math:`\text{gamma}^{\text{cycle iterations}}`
      at each cycle iteration.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_step (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_step=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.


    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    """
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: int,
        max_lr: int,
        step_size_up=2000,
        step_size_down=None,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
        cycle_momentum=True,
        base_momentum=0.8,
        max_momentum=0.9,
        last_step: int = -1,
        verbose: bool = False,
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = (
            step_size_down if step_size_down is not None else step_size_up
        )
        self.step_size_total = self.step_size_up + self.step_size_down
        self.step_up_ratio = step_size_up / self.step_size_total
        self.mode = mode
        self.gamma = gamma

        if mode not in ("triangular", "triangular2", "exp_range") and scale_fn is None:
            raise ValueError("mode is invalid and scale_fn is None")

        if scale_fn is None:
            if mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif mode == "exp_range":
                self.scale_fn: Callable[[int], float] = lambda x: gamma ** x
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if "momentum" not in optimizer._default_options:
                raise ValueError(
                    "optimizer must support momentum with `cycle_momentum` option enabled"
                )

            base_momentums = self._format_param(
                "base_momentum", optimizer, base_momentum
            )
            if last_step == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group["momentum"] = momentum
            self.base_momentums = [
                group["momentum"] for group in optimizer.param_groups
            ]
            self.max_momentums = self._format_param(
                "max_momentum", optimizer, max_momentum
            )

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
        offset_ratio = step % self.step_size_total / self.step_size_total
        if offset_ratio <= self.step_up_ratio:
            scale_factor = offset_ratio / self.step_up_ratio
        else:
            scale_factor = (1 - offset_ratio) / (1 - self.step_up_ratio)
        height = (self.max_lr - self.base_lr) * scale_factor

        if self.scale_mode == "cycle":
            cycle_num = 1.0 + step // self.step_size_total
            lr = self.base_lr + height * self.scale_fn(cycle_num)
        else:
            lr = self.base_lr + height * self.scale_fn(step)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(
                self.base_momentums, self.max_momentums
            ):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == "cycle":
                    momentum = max_momentum - base_height * self.scale_fn(cycle_num)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(
                        self.last_step
                    )
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group["momentum"] = momentum

        return lr

    def _generate_conf_for_graph(self, lr_conf):
        lr_conf.cyclic_lr_conf.SetInParent()
        cyclic_lr_conf = lr_conf.cyclic_lr_conf
        cyclic_lr_conf.base_lr = self.base_lr
        cyclic_lr_conf.max_lr = self.max_lr
        cyclic_lr_conf.step_size_up = self.step_size_up
        cyclic_lr_conf.step_size_down = self.step_size_down
        cyclic_lr_conf.mode = self.mode
        cyclic_lr_conf.gamma = self.gamma
        cyclic_lr_conf.scale_fn = self.scale_fn
        cyclic_lr_conf.scale_mode = self.scale_mode
        cyclic_lr_conf.cycle_momentum = self.cycle_momentum
        cyclic_lr_conf.base_momentum = self.base_momentums
        cyclic_lr_conf.max_momentum = self.max_momentums
