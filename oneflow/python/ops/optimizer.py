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
from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.remote_blob as remote_blob_util


class LrScheduler:
    def __init__(self, base_lr, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.warmup_mode = warmup_mode

    def ToProto(self):
        raise NotImplementedError()


class CosineScheduler(LrScheduler):
    def __init__(self, total_steps, base_lr, alpha=0.0, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.total_steps = total_steps
        self.alpha = alpha


def PiecewiseConstantScheduler(LrScheduler):
    def __init__(self, base_lr, boundaries, values, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.boundaries = boundaries
        self.values = values


def PiecewiseScalingScheduler(LrScheduler):
    def __init__(self, base_lr, boundaries, scales, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.boundaries = boundaries
        self.scales = scales


def PolynomialSchduler(LrScheduler):
    def __init__(self, base_lr, steps, end_learning_rate=0.0001, power=1.0, cycle=False, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.steps = steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle


class LinearConsineScheduler(LrScheduler):
    def __init__(self, total_steps, base_lr, num_periods=0.5, alpha=0.0, beta=0.001, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.total_steps = total_steps
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta


class ExponentialScheduler(LrScheduler):
    def __init__(self, steps, base_lr, decay_rate, staircase=False, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase


class InverseTimeScheduler(LrScheduler):
    def __init__(self, steps, base_lr, decay_rate, staircase=False, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase


class NaturalExpScheduler(LrScheduler):
    def __init__(self, steps, base_lr, decay_rate, staircase=False, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase


class WeightDecayConf:
    def __init__(self, rate, includes=None, excludes=None):
        pass


class Optimizer:
    def __init__(self, lr_scheduler: LrScheduler, weight_decay: Optional[Union[float, WeightDecayConf]], clip: Optional[Union[float, ClipConf]]):
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.clip = clip


class Sgd(Optimizer):
    def __init__(self, momentum)


flow.optimizer.Sgd(
    WarmupScheduler(CosineLrScheduler(total_steps=10000), warmup_steps=1000),
    weight_decay=1e-5,
    momentum=0.9
).minimize(loss)
