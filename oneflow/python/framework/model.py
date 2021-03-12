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

__all__ = ["CheckpointConfig", "Callback", "NumpyDataModule", "Model"]

from abc import ABC
from typing import Optional, Any, Union, Tuple

import inspect
import numpy as np

from oneflow.python.framework.check_point_v2 import (
    LoadVariables,
    SaveVarDict,
    GetCheckpoint,
)
from oneflow.python.framework.function_util import api_oneflow_function
from oneflow.python.framework.local_blob import LocalBlob
from oneflow.python.framework.module import Module
from oneflow.python.framework.session_util import api_clear_default_session
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.ops.optimizer import Optimizer
import oneflow.python.framework.typing as oneflow_typing
import oneflow.python.framework.dtype as dtype_util


@oneflow_export("model.CheckpointConfig")
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


@oneflow_export("model.Callback")
class Callback(ABC):
    r""" Abstract base class used to build new callbacks.
    """

    def on_training_step_end(
        self,
        outputs: Optional[Union[LocalBlob, Tuple[LocalBlob, ...]]],
        step_idx: int = 0,
        optimizer_idx: int = 0,
    ):
        # Do nothing, to be overrided by subclass.
        pass

    def on_validation_step_end(
        self,
        outputs: Optional[Union[LocalBlob, Tuple[LocalBlob, ...]]],
        step_idx: int = 0,
    ):
        # Do nothing, to be overrided by subclass.
        pass


@oneflow_export("model.DataModule")
class DataModule(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        # Do nothing, to be overrided by subclass.
        pass

    def inspect_data4model_construct(
        self, batch: Tuple[Any] = None, optimizer_idx: int = 0
    ):
        return None


@oneflow_export("model.NumpyDataModule")
class NumpyDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, step_idx: int = 0, optimizer_idx: int = 0):
        # Do nothing, to be overrided by subclass.
        pass

    def inspect_data4model_construct(
        self, batch: Tuple[np.ndarray, ...] = None, optimizer_idx: int = 0
    ):
        assert isinstance(batch, tuple), "model.NumpyDataModule must return a tuple."
        para_list = []
        for i, item in enumerate(batch):
            assert isinstance(item, np.ndarray)
            of_dtype = dtype_util.convert_numpy_dtype_to_oneflow_dtype(item.dtype)
            numpy_placeholder = oneflow_typing.Numpy.Placeholder(
                shape=item.shape, dtype=of_dtype
            )
            para_name = (
                self.__class__.__name__
                + "_opt_"
                + str(optimizer_idx)
                + "_para_"
                + str(i)
            )
            para_list.append(
                inspect.Parameter(
                    name=para_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=numpy_placeholder,
                )
            )
        return para_list


@oneflow_export("Model", "model.Model")
class Model(
    ABC, Module,
):
    r"""A high level API for model training and validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._is_deprecated_function_style = (
            kwargs["is_deprecated_function_style"]
            if "is_deprecated_function_style" in kwargs
            else False
        )
        if not self._is_deprecated_function_style:
            raise NotImplementedError

        self._training_config = (
            kwargs["training_config"] if "training_config" in kwargs else None
        )
        self._validation_config = (
            kwargs["validation_config"] if "validation_config" in kwargs else None
        )
        self._callbacks = kwargs["callbacks"] if "callbacks" in kwargs else []

        self._need_training = False
        self._training_is_numpy_input = False
        self._need_validation = False
        self._validation_is_numpy_input = False
        self._need_load_checkpoint = False
        self._need_save_checkpoint = False

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
        raise NotImplementedError()

    def configure_optimizers(self):
        r"""Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        """
        raise NotImplementedError()

    def fit(
        self,
        training_data: Optional[Union[DataModule, NumpyDataModule]] = None,
        validation_data: Optional[Union[DataModule, NumpyDataModule]] = None,
        validation_interval: int = 1,
        checkpoint_config: Optional[CheckpointConfig] = None,
        max_steps: int = 100,
    ):
        r""" Runs the full training and validation routine.
        """
        api_clear_default_session()

        self._max_steps = max_steps
        self._training_data = training_data
        self._validation_data = validation_data
        self._validation_interval = validation_interval
        self._checkpoint_config = checkpoint_config

        optim_conf = self.configure_optimizers()
        if isinstance(optim_conf, Optimizer):
            self._optimizers = [optim_conf]
        elif isinstance(optim_conf, (list, tuple)):
            self._optimizers = optim_conf

        self._train_jobs = []

        if self._method_overrided("training_step") and self._training_data is not None:
            self._need_training = True
            assert self._training_config is not None, "training_config cannot be None"
            self._training_is_numpy_input = (
                True if isinstance(self._training_data, NumpyDataModule) else False
            )

        if (
            self._method_overrided("validation_step")
            and self._validation_data is not None
        ):
            self._need_validation = True
            assert (
                self._validation_config is not None
            ), "validation_config cannot be None"
            self._validation_is_numpy_input = (
                True if isinstance(self._validation_data, NumpyDataModule) else False
            )

        if (
            self._checkpoint_config is not None
            and self._checkpoint_config.load_dirpath is not None
        ):
            self._need_load_checkpoint = True

        if (
            self._checkpoint_config is not None
            and self._checkpoint_config.save_dirpath is not None
        ):
            self._need_save_checkpoint = True

        if self._need_training:
            if not self._training_is_numpy_input:
                self._construct_train_jobs()
            else:
                self._numpy_input_first_train_batch = []
                for optimizer_idx in range(0, len(self._optimizers)):
                    batch = self._training_data(0, optimizer_idx)
                    self._construct_numpy_input_train_job(batch, optimizer_idx)
                    self._numpy_input_first_train_batch.insert(optimizer_idx, batch)

        if self._need_validation:
            if not self._validation_is_numpy_input:
                self._construct_eval_job()
            else:
                batch = self._validation_data(0)
                self._construct_numpy_input_eval_job(batch)
                self._numpy_input_first_val_batch = batch

        if self._need_load_checkpoint:
            self._load_checkpoint(dirpath=self._checkpoint_config.load_dirpath)

        for step_idx in range(0, self._max_steps):
            if self._need_training:
                for optimizer_idx in range(0, len(self._optimizers)):
                    outputs = None
                    if self._training_is_numpy_input:
                        batch = None
                        if step_idx == 0:
                            batch = self._numpy_input_first_train_batch[optimizer_idx]
                        else:
                            batch = self._training_data(step_idx, optimizer_idx)
                        outputs = self._train_jobs[optimizer_idx](*batch).get()
                    else:
                        # TODO(strint): Same job call for op data & numpy data.
                        #               If input data is blob, merge data graph into compute job
                        outputs = self._train_jobs[optimizer_idx]().get()
                    self._method_callback(
                        "on_training_step_end",
                        outputs=outputs,
                        step_idx=step_idx,
                        optimizer_idx=optimizer_idx,
                    )

            if self._need_validation:
                if (step_idx + 1) % self._validation_interval == 0:
                    eval_outputs = None
                    if self._validation_is_numpy_input:
                        batch = None
                        if step_idx == 0:
                            batch = self._numpy_input_first_val_batch
                        else:
                            batch = self._validation_data(step_idx)
                        eval_outputs = self._eval_job(*batch).get()
                    else:
                        # TODO(strint): Same job call for op data & numpy data
                        #               If input data is blob, merge data graph into compute job
                        eval_outputs = self._eval_job().get()
                    self._method_callback(
                        "on_validation_step_end",
                        step_idx=step_idx,
                        outputs=eval_outputs,
                    )

            if self._need_save_checkpoint:
                if (step_idx + 1) % self._checkpoint_config.save_interval == 0:
                    self._save_checkpoint(
                        dirpath=self._checkpoint_config.save_dirpath
                        + "-"
                        + str(step_idx)
                    )

    def _construct_one_train_job(self, optimizer_idx: int = 0):
        def job():
            batch = self._training_data()
            outputs = self.training_step(batch=batch, optimizer_idx=optimizer_idx)
            loss = None
            if isinstance(outputs, tuple) and len(outputs) > 0:
                loss = outputs[0]
            else:
                loss = outputs
            self._optimizers[optimizer_idx].minimize(loss)
            return outputs

        job.__name__ = (
            self.__class__.__name__ + "_Model_train_job_" + str(optimizer_idx)
        )
        deco = api_oneflow_function(type="train", function_config=self._training_config)
        return deco(job)

    def _construct_train_jobs(self):
        if len(self._optimizers) == 1:
            self._train_jobs.append(self._construct_one_train_job(0))
        else:
            for optimizer_idx in range(0, len(self._optimizers)):
                self._train_jobs.append(self._construct_one_train_job(optimizer_idx))

    def _construct_eval_job(self):
        def job():
            batch = self._validation_data()
            return self.validation_step(batch)

        job.__name__ = self.__class__.__name__ + "_Model_eval_job"
        deco = api_oneflow_function(
            type="predict", function_config=self._validation_config
        )
        self._eval_job = deco(job)

    def _construct_numpy_input_train_job(self, batch, optimizer_idx):
        def job(*input_batch):
            outputs = self.training_step(batch=input_batch, optimizer_idx=optimizer_idx)
            loss = None
            if isinstance(outputs, tuple) and len(outputs) > 0:
                loss = outputs[0]
            else:
                loss = outputs
            self._optimizers[optimizer_idx].minimize(loss)
            return outputs

        para_list = self._training_data.inspect_data4model_construct(
            batch, optimizer_idx
        )

        origin_sig = inspect.signature(job)
        new_sig = origin_sig.replace(parameters=para_list)
        job.__oneflow_function_signature__ = new_sig
        job.__name__ = (
            self.__class__.__name__ + "_Model_train_numpy_job_" + str(optimizer_idx)
        )
        deco = api_oneflow_function(type="train", function_config=self._training_config)
        train_job = deco(job)
        self._train_jobs.insert(optimizer_idx, train_job)

    def _construct_numpy_input_eval_job(self, batch: Tuple[np.ndarray, ...] = None):
        def job(*input_batch):
            return self.validation_step(batch=input_batch)

        para_list = self._validation_data.inspect_data4model_construct(batch, 0)
        origin_sig = inspect.signature(job)
        new_sig = origin_sig.replace(parameters=para_list)
        job.__oneflow_function_signature__ = new_sig
        job.__name__ = self.__class__.__name__ + "_Model_eval_numpy_job"
        deco = api_oneflow_function(
            type="predict", function_config=self._validation_config
        )
        self._eval_job = deco(job)

    def _save_checkpoint(
        self, dirpath: str,
    ):
        r"""Save model states as a checkpoint.
        """
        SaveVarDict(path=dirpath)

    def _load_checkpoint(
        self, dirpath: str,
    ):
        r"""Load model states from a checkpoint.
        """
        LoadVariables(GetCheckpoint(path=dirpath))

    def _method_overrided(self, method_name: str = None) -> bool:
        return getattr(self.__class__, method_name) != getattr(Model, method_name)

    def _method_callback(self, method_name: str = None, *args, **kwargs):
        for cb in self._callbacks:
            method = getattr(cb, method_name)
            method(*args, **kwargs)
