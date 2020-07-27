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

import oneflow as flow
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
from typing import Tuple, Optional, Union, Sequence


class LrScheduler:
    def __init__(
        self,
        base_lr: float,
        lr_lbn: Optional[str] = None,
        warmup_steps: int = 0,
        warmup_begin_multiplier: float = 0,
        warmup_mode: str = "linear",
    ):
        assert warmup_mode in ["linear", "constant"]
        self.base_lr = base_lr
        self.lr_lbn = lr_lbn
        self.warmup_steps = warmup_steps
        self.warmup_begin_multiplier = warmup_begin_multiplier
        self.warmup_mode = warmup_mode

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        raise NotImplementedError()

    def SetLrFieldsInTrainConf(self, train_conf) -> None:
        if self.warmup_conf is not None:
            train_conf.model_update_conf.warmup_conf.CopyFrom(self.warmup_conf)
        if self.lr_lbn is not None:
            assert self.learning_rate_decay_conf is None
            train_conf.primary_lr_lbn = self.lr_lbn
            # primary_lr is a required field
            train_conf.primary_lr = self.base_lr
        else:
            assert self.learning_rate_decay_conf is not None
            train_conf.model_update_conf.learning_rate_decay.CopyFrom(
                self.learning_rate_decay_conf
            )
            train_conf.primary_lr = self.base_lr

    @property
    def warmup_conf(self) -> op_conf_pb.WarmupConf:
        if self.warmup_steps == 0:
            return None
        warmup_conf = op_conf_pb.WarmupConf()
        if self.warmup_mode == "linear":
            warmup_conf.linear_conf.warmup_batches = self.warmup_steps
            warmup_conf.linear_conf.start_multiplier = self.warmup_begin_multiplier
        elif self.warmup_mode == "constant":
            # TODO(daquexian):
            pass
        else:
            raise RuntimeError()
        return warmup_conf


@oneflow_export("optimizer.CosineScheduler")
class CosineScheduler(LrScheduler):
    def __init__(
        self,
        total_steps: int,
        base_lr: float,
        alpha: float = 0.0,
        warmup_steps: int = 0,
        warmup_begin_multiplier: float = 0,
        warmup_mode: str = "linear",
    ):
        super().__init__(
            base_lr,
            warmup_steps=warmup_steps,
            warmup_begin_multiplier=warmup_begin_multiplier,
            warmup_mode=warmup_mode,
        )
        self.total_steps = total_steps
        self.alpha = alpha

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.cosine_conf.decay_batches = self.total_steps
        learning_rate_decay_conf.cosine_conf.alpha = self.alpha
        return learning_rate_decay_conf


@oneflow_export("optimizer.CustomScheduler")
class CustomScheduler(LrScheduler):
    def __init__(self, lbn: str):
        super().__init__(0, lr_lbn=lbn)

    @property
    def learning_rate_decay_conf(self) -> op_conf_pb.LearningRateDecayConf:
        return None


class Optimizer:
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        loss_scale_factor: Optional[float] = None,
        weight_decay: Optional[float] = None,
        weight_decay_includes: Optional[Sequence[str]] = None,
        weight_decay_excludes: Optional[Sequence[str]] = None,
        clip: Optional[float] = None,
        train_step_lbn: Optional[str] = None,
    ):
        self.lr_scheduler = lr_scheduler
        self.loss_scale_factor = loss_scale_factor
        self.weight_decay = weight_decay
        self.weight_decay_includes = weight_decay_includes
        self.weight_decay_excludes = weight_decay_excludes
        self.clip = clip
        self.train_step_lbn = train_step_lbn

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        raise NotImplementedError()

    @property
    def train_conf(self) -> job_conf_pb.TrainConf:
        train_conf = job_conf_pb.TrainConf()
        self.lr_scheduler.SetLrFieldsInTrainConf(train_conf)
        update_conf = train_conf.model_update_conf
        if self.clip is not None:
            update_conf.clip_conf.clip_by_global_norm.clip_norm = self.clip
        if self.weight_decay is not None:
            update_conf.weight_decay_conf.weight_decay_rate = weight_decay
            assert not (
                self.weight_decay_excludes is not None
                and self.weight_decay_includes is not None
            )
            # TODO(daquexian):
            if self.weight_decay_includes is not None:
                pass
            else:
                pass
        if self.train_step_lbn is not None:
            train_conf.train_step_lbn = self.train_step_lbn
        self._SetSpecificFieldsInTrainConf(train_conf)
        return train_conf

    def minimize(self, loss) -> None:
        c_api_util.CurJobBuildAndInferCtx_SetTrainConf(self.train_conf)
        flow.losses.add_loss(loss)


@oneflow_export("optimizer.SGD")
class SGD(Optimizer):
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        loss_scale_factor: Optional[float] = None,
        momentum: int = 0.9,
        weight_decay: Optional[float] = None,
        weight_decay_includes: Optional[str] = None,
        weight_decay_excludes: Optional[str] = None,
        clip: Optional[float] = None,
        train_step_lbn: Optional[str] = None,
    ):
        super().__init__(
            lr_scheduler,
            loss_scale_factor,
            weight_decay,
            weight_decay_includes,
            weight_decay_excludes,
            clip,
            train_step_lbn,
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
        weight_decay: Optional[float] = None,
        weight_decay_includes: Optional[str] = None,
        weight_decay_excludes: Optional[str] = None,
        clip: Optional[float] = None,
        train_step_lbn: Optional[str] = None,
    ):
        super().__init__(
            lr_scheduler,
            loss_scale_factor,
            weight_decay,
            weight_decay_includes,
            weight_decay_excludes,
            clip,
            train_step_lbn,
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
