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

import collections.abc

import oneflow as flow
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
from typing import Tuple, Optional, Union, Sequence, Text


class ClipGradientConf:
    @property
    def clip_conf(self) -> op_conf_pb.ClipConf:
        raise NotImplementedError()


@oneflow_export("optimizer.grad_clipping.by_global_norm")
class ClipByGlobalNorm(ClipGradientConf):
    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    @property
    def clip_conf(self):
        clip_conf = op_conf_pb.ClipConf()
        clip_conf.clip_by_global_norm.clip_norm = self.clip_norm
        return clip_conf


class WarmupConf:
    @property
    def warmup_conf(self) -> op_conf_pb.WarmupConf:
        raise NotImplementedError()


@oneflow_export("optimizer.warmup.constant")
class ConstantWarmup(WarmupConf):
    def __init__(self, steps, multiplier):
        self.steps = steps
        self.multiplier = multiplier

    @property
    def warmup_conf(self) -> op_conf_pb.WarmupConf:
        warmup_conf = op_conf_pb.WarmupConf()
        warmup_conf.constant_conf.warmup_batches = self.steps
        warmup_conf.constant_conf.multiplier = self.multiplier
        return warmup_conf


@oneflow_export("optimizer.warmup.linear")
class LinearWarmup(WarmupConf):
    def __init__(self, steps, start_multiplier):
        self.steps = steps
        self.start_multiplier = start_multiplier

    @property
    def warmup_conf(self) -> op_conf_pb.WarmupConf:
        warmup_conf = op_conf_pb.WarmupConf()
        warmup_conf.linear_conf.warmup_batches = self.steps
        warmup_conf.linear_conf.start_multiplier = self.start_multiplier
        return warmup_conf


class LrScheduler:
    def __init__(
        self,
        base_lr: Optional[float] = None,
        lr_lbn: Optional[Text] = None,
        warmup: Optional[WarmupConf] = None,
    ):
        self.base_lr = base_lr
        self.lr_lbn = lr_lbn
        self.warmup = warmup

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        raise NotImplementedError()

    def SetLrFieldsInTrainConf(self, train_conf) -> None:
        if self.warmup_conf is not None:
            train_conf.model_update_conf.warmup_conf.CopyFrom(self.warmup_conf)
        if self.lr_lbn is not None:
            assert self.learning_rate_decay_conf is None
            assert self.base_lr is None
            train_conf.primary_lr_lbn = self.lr_lbn
            # primary_lr is a required field
            train_conf.primary_lr = 0
        else:
            assert self.learning_rate_decay_conf is not None
            train_conf.model_update_conf.learning_rate_decay.CopyFrom(
                self.learning_rate_decay_conf
            )
            train_conf.primary_lr = self.base_lr

    @property
    def warmup_conf(self) -> op_conf_pb.WarmupConf:
        if self.warmup is None:
            return None
        return self.warmup.warmup_conf


@oneflow_export("optimizer.CosineScheduler")
class CosineScheduler(LrScheduler):
    def __init__(
        self,
        base_lr: float,
        steps: int,
        alpha: float = 0.0,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.alpha = alpha

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.cosine_conf.decay_batches = self.steps
        learning_rate_decay_conf.cosine_conf.alpha = self.alpha
        return learning_rate_decay_conf


@oneflow_export("optimizer.CustomScheduler")
class CustomScheduler(LrScheduler):
    def __init__(self, lbn: Text):
        super().__init__(lr_lbn=lbn)

    @property
    def learning_rate_decay_conf(self) -> op_conf_pb.LearningRateDecayConf:
        return None


@oneflow_export("optimizer.PiecewiseConstantScheduler")
class PiecewiseConstantScheduler(LrScheduler):
    def __init__(
        self,
        boundaries: Sequence[int],
        values: Sequence[float],
        warmup: Optional[WarmupConf] = None,
    ):
        assert len(boundaries) + 1 == len(values)
        super().__init__(base_lr=values[0], warmup=warmup)
        self.boundaries = boundaries
        self.values = values

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.piecewise_constant_conf.boundaries.extend(
            self.boundaries
        )
        learning_rate_decay_conf.piecewise_constant_conf.values.extend(self.values)
        return learning_rate_decay_conf


@oneflow_export("optimizer.PiecewiseScalingScheduler")
class PiecewiseScalingScheduler(LrScheduler):
    def __init__(
        self,
        base_lr: float,
        boundaries: Sequence[int],
        scale: Union[float, Sequence[float]],
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.boundaries = boundaries
        if not isinstance(scale, collections.abc.Sequence):
            scale = [scale] * len(boundaries)
        assert len(boundaries) == len(scale)
        self.scale = [1] + list(scale)

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.piecewise_scaling_conf.boundaries.extend(
            self.boundaries
        )
        learning_rate_decay_conf.piecewise_scaling_conf.scales.extend(self.scale)
        return learning_rate_decay_conf


@oneflow_export("optimizer.PolynomialSchduler")
class PolynomialSchduler(LrScheduler):
    def __init__(
        self,
        base_lr: float,
        steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.polynomial_conf.decay_batches = self.steps
        learning_rate_decay_conf.polynomial_conf.end_learning_rate = (
            self.end_learning_rate
        )
        learning_rate_decay_conf.polynomial_conf.power = self.power
        learning_rate_decay_conf.polynomial_conf.cycle = self.cycle
        return learning_rate_decay_conf


@oneflow_export("optimizer.LinearCosineScheduler")
class LinearCosineScheduler(LrScheduler):
    def __init__(
        self,
        base_lr: float,
        steps: int,
        num_periods: float = 0.5,
        alpha: float = 0.0,
        beta: float = 0.001,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.linear_cosine_conf.decay_batches = self.steps
        learning_rate_decay_conf.linear_cosine_conf.num_periods = self.num_periods
        learning_rate_decay_conf.polynomial_conf.alpha = self.alpha
        learning_rate_decay_conf.polynomial_conf.beta = self.beta
        return learning_rate_decay_conf


@oneflow_export("optimizer.ExponentialScheduler")
class ExponentialScheduler(LrScheduler):
    def __init__(
        self,
        base_lr: float,
        steps: int,
        decay_rate: float,
        staircase=False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.exponential_conf.decay_batches = self.steps
        learning_rate_decay_conf.exponential_conf.decay_rate = self.decay_rate
        learning_rate_decay_conf.exponential_conf.staircase = self.staircase
        return learning_rate_decay_conf


@oneflow_export("optimizer.InverseTimeScheduler")
class InverseTimeScheduler(LrScheduler):
    def __init__(
        self,
        base_lr: float,
        steps: int,
        decay_rate: float,
        staircase: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.inverse_time_conf.decay_batches = self.steps
        learning_rate_decay_conf.inverse_time_conf.decay_rate = self.decay_rate
        learning_rate_decay_conf.inverse_time_conf.staircase = self.staircase
        return learning_rate_decay_conf


@oneflow_export("optimizer.NaturalExpScheduler")
class NaturalExpScheduler(LrScheduler):
    def __init__(
        self,
        base_lr: float,
        steps: int,
        decay_rate: float,
        staircase: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.natural_exp_conf.decay_batches = self.steps
        learning_rate_decay_conf.natural_exp_conf.decay_rate = self.decay_rate
        learning_rate_decay_conf.natural_exp_conf.staircase = self.staircase
        return learning_rate_decay_conf


class Optimizer:
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        loss_scale_factor: Optional[int] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        self.lr_scheduler = lr_scheduler
        self.loss_scale_factor = loss_scale_factor
        self.grad_clipping = grad_clipping
        self.train_step_lbn = train_step_lbn

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        raise NotImplementedError()

    @property
    def train_conf(self) -> job_conf_pb.TrainConf:
        train_conf = job_conf_pb.TrainConf()
        self.lr_scheduler.SetLrFieldsInTrainConf(train_conf)
        update_conf = train_conf.model_update_conf
        if self.grad_clipping is not None:
            update_conf.clip_conf.CopyFrom(self.grad_clipping.clip_conf)
        if self.train_step_lbn is not None:
            train_conf.train_step_lbn = self.train_step_lbn
        if self.loss_scale_factor is not None:
            update_conf.loss_scale_factor = self.loss_scale_factor
        self._SetSpecificFieldsInTrainConf(train_conf)
        return train_conf

    def minimize(
        self, loss: Union[Sequence[remote_blob_util.BlobDef], remote_blob_util.BlobDef]
    ) -> None:
        if not isinstance(loss, collections.abc.Sequence):
            loss = [loss]
        c_api_util.CurJobBuildAndInferCtx_SetTrainConf(self.train_conf)
        for x in loss:
            flow.losses.add_loss(x)


@oneflow_export("optimizer.SGD")
class SGD(Optimizer):
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        loss_scale_factor: Optional[float] = None,
        momentum: int = 0.9,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.momentum = momentum

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        if self.momentum == 0:
            train_conf.model_update_conf.naive_conf.SetInParent()
        else:
            train_conf.model_update_conf.momentum_conf.beta = self.momentum


@oneflow_export("optimizer.Adam")
class Adam(Optimizer):
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        do_bias_correction=False,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.do_bias_correction = do_bias_correction

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.adam_conf.beta1 = self.beta1
        train_conf.model_update_conf.adam_conf.beta2 = self.beta2
        train_conf.model_update_conf.adam_conf.epsilon = self.epsilon
        train_conf.model_update_conf.adam_conf.do_bias_correction = (
            self.do_bias_correction
        )


@oneflow_export("optimizer.AdamW")
class AdamW(Optimizer):
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        do_bias_correction=False,
        loss_scale_factor: Optional[float] = None,
        weight_decay: Optional[float] = None,
        weight_decay_includes: Optional[Union[Sequence[Text], Text]] = None,
        weight_decay_excludes: Optional[Union[Sequence[Text], Text]] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.do_bias_correction = do_bias_correction
        self.weight_decay = weight_decay
        if isinstance(weight_decay_includes, str):
            weight_decay_includes = [weight_decay_includes]
        if isinstance(weight_decay_excludes, str):
            weight_decay_excludes = [weight_decay_excludes]
        self.weight_decay_includes = weight_decay_includes
        self.weight_decay_excludes = weight_decay_excludes

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.adam_conf.beta1 = self.beta1
        train_conf.model_update_conf.adam_conf.beta2 = self.beta2
        train_conf.model_update_conf.adam_conf.epsilon = self.epsilon
        train_conf.model_update_conf.adam_conf.do_bias_correction = (
            self.do_bias_correction
        )
        if self.weight_decay is not None:
            train_conf.model_update_conf.weight_decay_conf.weight_decay_rate = (
                self.weight_decay
            )
            assert not (
                self.weight_decay_excludes is not None
                and self.weight_decay_includes is not None
            )
            if self.weight_decay_includes is not None:
                train_conf.model_update_conf.weight_decay_conf.includes.pattern.extend(
                    self.weight_decay_includes
                )
            elif self.weight_decay_excludes is not None:
                train_conf.model_update_conf.weight_decay_conf.excludes.pattern.extend(
                    self.weight_decay_excludes
                )


@oneflow_export("optimizer.RMSProp")
class RMSProp(Optimizer):
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        decay_rate: float = 0.99,
        epsilon: float = 1e-8,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.decay_rate = decay_rate
        self.epsilon = epsilon

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.rmsprop_conf.decay_rate = self.decay_rate
        train_conf.model_update_conf.rmsprop_conf.epsilon = self.epsilon


@oneflow_export("optimizer.LARS")
class LARS(Optimizer):
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        momentum_beta: float = 0.9,
        epsilon: float = 1e-9,
        lars_coefficient: float = 0.0001,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.momentum_beta = momentum_beta
        self.epsilon = epsilon
        self.lars_coefficient = lars_coefficient

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.lars_conf.momentum_beta = self.momentum_beta
        train_conf.model_update_conf.lars_conf.epsilon = self.epsilon
        train_conf.model_update_conf.lars_conf.lars_coefficient = self.lars_coefficient


@oneflow_export("optimizer.LazyAdam")
class LazyAdam(Optimizer):
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.lazy_adam_conf.beta1 = self.beta1
        train_conf.model_update_conf.lazy_adam_conf.beta2 = self.beta2
        train_conf.model_update_conf.lazy_adam_conf.epsilon = self.epsilon
