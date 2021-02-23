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
from abc import ABC
from typing import Optional

from oneflow.python.framework.check_point_v2 import *
from oneflow.python.framework.function_util import api_oneflow_function
from oneflow.python.framework.module import Module
from oneflow.python.framework.session_util import api_clear_default_session
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.ops.optimizer import Optimizer


@oneflow_export("ModelCheckpointConfig")
class CheckpointConfig(object):
    def __init__(
        self,
        load_dirpath: str = None,
        save_dirpath: str = None,
        save_interval: int = 1,
    ):
        self.load_dirpath = load_dirpath
        self.save_dirpath = save_dirpath
        self.save_interval = save_interval


@oneflow_export("ModelCallback")
class Callback(ABC):
    r""" Abstract base class used to build new callbacks.
    """

    def on_training_step_end(self, step, outputs):
        pass

    def on_validation_step_end(self, step, outputs):
        pass


@oneflow_export("Model")
class Model(
    ABC, Module,
):
    r"""A high level API for model training and validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.is_function_style = kwargs["is_function_style"]
        self.training_config = kwargs["training_config"]
        self.validation_config = kwargs["validation_config"]
        self.callbacks = kwargs["callbacks"]

        optim_conf = self.configure_optimizers()
        if isinstance(optim_conf, Optimizer):
            self.optimizers = [optim_conf]
        elif isinstance(optim_conf, (list, tuple)):
            self.optimizers = optim_conf

        self.need_training = False
        self.need_validation = False
        self.need_checkpoint = False

    def forward(self, *args, **kwargs):
        r"""Same as `nn.Module.forward()`, here is to define the operations you want to use for prediction.
        """
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        r"""Operates on a single batch of data from the training set and return loss.
        """
        raise NotImplementedError()

    def validation_step(self, *args, **kwargs):
        r"""Operates on a single batch of data from the validation set.
        """

    def configure_optimizers(self):
        r"""Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        """
        raise NotImplementedError()

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
    ) -> None:
        r"""Customized optimizer action.
        """

    def _func_train_job(self):
        @api_oneflow_function(type="train", function_config=self.training_config)
        def job():
            batch = self.training_data()
            loss = self.training_step(batch)
            self.optimizers[0].minimize(loss)
            return loss

        return job

    def _func_eval_job(self):
        @api_oneflow_function(function_config=self.validation_config)
        def eval_job():
            batch = self.validation_data()
            return self.validation_step(batch)

        return eval_job

    def fit(
        self,
        training_data=None,
        validation_data=None,
        validation_interval: int = 1,
        checkpoint_config=None,
        max_steps: int = 100,
    ):
        api_clear_default_session()

        self.max_steps = max_steps
        self.training_data = training_data
        self.validation_data = validation_data
        self.validation_interval = validation_interval
        self.checkpoint_config = checkpoint_config

        if self._method_overrided("training_step") and self.training_data is not None:
            self.need_training = True
            self.train_job = self._func_train_job()

        if (
            self._method_overrided("validation_step")
            and self.validation_data is not None
        ):
            self.need_validation = True
            self.eval_job = self._func_eval_job()

        if self.checkpoint_config.load_dirpath is not None:
            self.load_checkpoint(dirpath=self.checkpoint_config.load_dirpath)

        if self.checkpoint_config.save_dirpath is not None:
            self.need_checkpoint = True

        for step in range(0, self.max_steps):
            if self.need_training:
                loss = self.train_job().get()
                self._method_callback("on_training_step_end", step, loss)
            if self.need_validation:
                if (step + 1) % self.validation_interval == 0:
                    eval_loss = self.eval_job().get()
                    self._method_callback("on_validation_step_end", step, eval_loss)
            if self.need_checkpoint:
                if (step + 1) % self.checkpoint_config.save_interval == 0:
                    self.save_checkpoint(
                        dirpath=self.checkpoint_config.save_dirpath + "-" + str(step)
                    )

    def save_checkpoint(
        self, dirpath,
    ):
        r"""Save model states as a checkpoint.
        """
        SaveVarDict(path=dirpath)

    def load_checkpoint(
        self, dirpath,
    ):
        r"""Load model states from a checkpoint.
        """
        LoadVariables(GetCheckpoint(path=dirpath))

    def _method_overrided(self, method_name: str = None) -> bool:
        return getattr(self.__class__, method_name) != getattr(Model, method_name)

    def _method_callback(self, method_name: str = None, *args, **kwargs):
        for cb in self.callbacks:
            method = getattr(cb, method_name)
            method(*args, **kwargs)
