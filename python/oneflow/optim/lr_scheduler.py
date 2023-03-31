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
from oneflow.nn.optimizer.lr_scheduler import LRScheduler as _LRScheduler
from oneflow.nn.optimizer.cosine_decay_lr import CosineDecayLR
from oneflow.nn.optimizer.cosine_annealing_lr import CosineAnnealingLR
from oneflow.nn.optimizer.lambda_lr import LambdaLR
from oneflow.nn.optimizer.step_lr import StepLR
from oneflow.nn.optimizer.multistep_lr import MultiStepLR
from oneflow.nn.optimizer.exponential_lr import ExponentialLR
from oneflow.nn.optimizer.reduce_lr_on_plateau import ReduceLROnPlateau
from oneflow.nn.optimizer.polynomial_lr import PolynomialLR
from oneflow.nn.optimizer.constant_lr import ConstantLR
from oneflow.nn.optimizer.linear_lr import LinearLR
from oneflow.nn.optimizer.warmup_lr import WarmupLR
from oneflow.nn.optimizer.warmup_lr import WarmupLR as WarmUpLR
from oneflow.nn.optimizer.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestarts,
)
from oneflow.nn.optimizer.chained_scheduler import ChainedScheduler
from oneflow.nn.optimizer.sequential_lr import SequentialLR
from oneflow.nn.optimizer.multiplicative_lr import MultiplicativeLR
