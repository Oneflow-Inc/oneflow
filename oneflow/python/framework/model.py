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

__all__ = [
    "DataModule",
    "NumpyDataModule",
    "TrainingConfig",
    "ValidationConfig",
    "CheckpointConfig",
    "Callback",
    "Model",
]

from abc import ABC
from typing import Optional, Any, Union, Tuple, List

import inspect
import numpy as np

from oneflow.python.framework.check_point_v2 import (
    LoadVariables,
    SaveVarDict,
    GetCheckpoint,
)
from oneflow.python.framework.function_util import api_oneflow_function
from oneflow.python.framework.function_util import FunctionConfig as ExecutionConfig
from oneflow.python.framework.local_blob import LocalBlob
from oneflow.python.framework.module import Module as DeprecatedModule
from oneflow.python.framework.session_util import api_clear_default_session
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.ops.optimizer import Optimizer
import oneflow.python.framework.typing as oneflow_typing
import oneflow.python.framework.dtype as dtype_util


@oneflow_export("model.DataModule")
class DataModule(DeprecatedModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        # Do nothing, to be overrided by subclass.
        pass

    def infer_oneflow_data_placeholder(
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

    def infer_oneflow_data_placeholder(
        self, batch: Tuple[np.ndarray, ...] = None, optimizer_idx: int = 0
    ):
        assert isinstance(batch, tuple), "model.NumpyDataModule must return a tuple."
        data_placeholder_list = []
        for i, item in enumerate(batch):
            assert isinstance(
                item, np.ndarray
            ), "model.NumpyDataModule must return a tuple of numpy."
            of_dtype = dtype_util.convert_numpy_dtype_to_oneflow_dtype(item.dtype)
            numpy_placeholder = oneflow_typing.Numpy.Placeholder(
                shape=item.shape, dtype=of_dtype
            )
            data_placeholder_list.append(numpy_placeholder)
        return data_placeholder_list


@oneflow_export("model.TrainingConfig")
class TrainingConfig:
    def __init__(self):
        super().__init__()
        self.exe_cfg = None
        self.data = None
        self.error_msg = ""

    def config_execution(self, exe_cfg: ExecutionConfig = None):
        self.exe_cfg = exe_cfg

    def config_data(self, data: DataModule = None):
        self.data = data

    def check_valid(self):
        is_valid = True
        self.error_msg = ""
        if self.exe_cfg is None:
            self.error_msg += "model.TrainingConfig exe_cfg is None;"
            is_valid = False
        if not isinstance(self.exe_cfg, ExecutionConfig):
            self.error_msg += "model.TrainingConfig exe_cfg is not ExecutionConfig;"
            is_valid = False
        if self.data is None:
            self.error_msg += "model.TrainingConfig data is None;"
            is_valid = False
        if not isinstance(self.data, DataModule):
            self.error_msg += "model.TrainingConfig data is not DataModule;"
            is_valid = False
        return is_valid


@oneflow_export("model.ValidationConfig")
class ValidationConfig:
    def __init__(self):
        super().__init__()
        self.exe_cfg = None
        self.data = None
        self.step_interval = 10
        self.error_msg = ""

    def config_execution(self, exe_cfg: ExecutionConfig = None):
        self.exe_cfg = exe_cfg

    def config_data(self, data: DataModule = None):
        self.data = data

    def config_step_interval(self, step_interval: int = 1):
        self.step_interval = step_interval

    def check_valid(self):
        is_valid = True
        self.error_msg = ""
        if self.exe_cfg is None:
            self.error_msg += "model.ValidationConfig exe_cfg is None;"
            is_valid = False
        if not isinstance(self.exe_cfg, ExecutionConfig):
            self.error_msg += "model.ValidationConfig exe_cfg is not ExecutionConfig;"
            is_valid = False
        if self.data is None:
            self.error_msg += "model.ValidationConfig data is None;"
            is_valid = False
        if not isinstance(self.data, DataModule):
            self.error_msg += "model.ValidationConfig data is not DataModule;"
            is_valid = False
        if self.step_interval <= 0:
            self.error_msg += "model.ValidationConfig step_interval is <= 0;"
            is_valid = False
        return is_valid


@oneflow_export("model.CheckpointConfig")
class CheckpointConfig(object):
    def __init__(self,):
        self.need_load = False
        self.load_dirpath = None
        self.need_save = False
        self.save_dirpath = None
        self.save_step_interval = 1
        self.error_msg = ""

    def config_load(self, dirpath: str = None):
        self.need_load = True
        assert dirpath is not None, "dirpath should not be None"
        self.load_dirpath = dirpath

    def config_save(self, dirpath: str = None, step_interval: int = 1):
        self.need_save = True
        self.save_dirpath = dirpath
        assert dirpath is not None, "dirpath should not be None"
        self.save_step_interval = step_interval
        assert step_interval > 0, "step_interval should not <= 0"

    def check_valid(self):
        # Reserved interface for future use
        is_valid = True
        self.error_msg = ""
        return is_valid


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


@oneflow_export("Model", "model.Model")
class Model(
    ABC, DeprecatedModule,
):
    r"""A high level API for model training and validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._is_deprecated_function_style = False

        self._is_deprecated_function_style = (
            kwargs["is_deprecated_function_style"]
            if "is_deprecated_function_style" in kwargs
            else False
        )
        if not self._is_deprecated_function_style:
            raise NotImplementedError

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
        training_config: Optional[TrainingConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
        max_steps: int = 100,
    ):
        r""" Runs the full training and validation routine.
        """
        self._max_steps = max_steps
        api_clear_default_session()
        self._sub_models = self._get_and_check_sub_models(
            training_config, validation_config, checkpoint_config, callbacks
        )

        if len(self._sub_models) == 0:
            return

        if self._checkpoint_model.is_valid:
            self._checkpoint_model.load()
        for step_idx in range(0, self._max_steps):
            for sub_model in self._sub_models:
                try:
                    sub_model.step(step_idx)
                except Exception as e:
                    print(
                        "Model step_idx {} sub-model {} failed.".format(
                            step_idx, sub_model.name
                        )
                    )
                    raise e

    def method_overrided(self, method_name: str = None) -> bool:
        return getattr(self.__class__, method_name) != getattr(Model, method_name)

    def _get_and_check_sub_models(
        self,
        training_config: Optional[TrainingConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ):
        sub_models = []

        self._train_model = TrainModel(training_config, self, callbacks)
        if self._train_model.is_valid:
            sub_models.append(self._train_model)
        else:
            print(
                self._train_model.error_msg,
                " {}'s fit() will not do training.".format(self.__class__.__name__),
            )

        self._val_model = ValidateModel(validation_config, self, callbacks)
        if self._val_model.is_valid:
            sub_models.append(self._val_model)
        else:
            print(
                self._val_model.error_msg,
                " {}'s fit() will not do validation.".format(self.__class__.__name__),
            )

        if len(sub_models) == 0:
            print(" {}'s fit() will do nothing.".format(self.__class__.__name__))
            return sub_models

        self._checkpoint_model = CheckpointModel(checkpoint_config, self, callbacks)
        if self._checkpoint_model.is_valid:
            sub_models.append(self._checkpoint_model)
        else:
            print(
                self._checkpoint_model.error_msg,
                " {}'s fit() will not do checkpoint.".format(self.__class__.__name__),
            )

        return sub_models


class SubModel(ABC):
    def __init__(self, name, cfg, model, callbacks):
        self._cfg = cfg
        assert isinstance(model, Model)
        self._model = model
        self._cbs = callbacks

        self.name = name
        self.is_valid = True
        self.error_msg = self._model.__class__.__name__ + " " + self.name + " "

        if not self._get_and_check_cfg():
            self.is_valid = False

        if not self._get_and_check_cbs():
            self.is_valid = False

    def step(self, step_idx: int = 0):
        raise NotImplementedError

    def _get_and_check_cfg(self):
        if self._cfg is None:
            self.error_msg += "config is None;"
            return False

        if not self._cfg.check_valid():
            self.error_msg += self._cfg.error_msg
            return False
        else:
            return True

    def _get_and_check_cbs(self):
        if self._cbs is None:
            self._cbs = []
            return True

        if isinstance(self._cbs, Callback):
            self._cbs = [self._cbs]
            return True

        if isinstance(self._cbs, list):
            for cb in self._cbs:
                assert isinstance(
                    cb, Callback
                ), "model callbacks' type must be model.Callback or List[model.Callback]."
            return True

        assert (
            False
        ), "model callbacks' type must be model.Callback or List[model.Callback]."

    def _method_callback(self, method_name: str = None, *args, **kwargs):
        for cb in self._cbs:
            method = getattr(cb, method_name)
            method(*args, **kwargs)


class TrainModel(SubModel):
    def __init__(
        self,
        cfg: TrainingConfig = None,
        model: Model = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ):
        super().__init__("train_model", cfg, model, callbacks)

        if not self._get_and_check_step():
            self.is_valid = False

        if not self._get_and_check_opts():
            self.is_valid = False

        if self.is_valid and not self._get_and_check_jobs():
            self.is_valid = False

    def step(self, step_idx: int = 0):
        assert self.is_valid, self.error_msg
        for optimizer_idx in range(0, len(self._opts)):
            outputs = None
            if self._is_numpy_input:
                batch = None
                if step_idx == 0:
                    batch = self._first_numpy_batch[optimizer_idx]
                else:
                    batch = self._cfg.data(step_idx, optimizer_idx)
                outputs = self._jobs[optimizer_idx](*batch).get()
            else:
                outputs = self._jobs[optimizer_idx]().get()

            self._method_callback(
                "on_training_step_end",
                outputs=outputs,
                step_idx=step_idx,
                optimizer_idx=optimizer_idx,
            )

    def _get_and_check_step(self):
        if not self._model.method_overrided("training_step"):
            self.error_msg += "model.training_step() is empty;"
            return False
        else:
            return True

    def _get_and_check_opts(self):
        self._opts = []
        if not self._model.method_overrided("configure_optimizers"):
            self.error_msg += "model.configure_optimizers() is empty;"
            return False

        opt_conf = self._model.configure_optimizers()
        if isinstance(opt_conf, Optimizer):
            self._opts = [opt_conf]
        elif isinstance(opt_conf, (list, tuple)):
            for opt in opt_conf:
                assert isinstance(
                    opt, Optimizer
                ), "model.configure_optimizers() must return Optimizer "
                "or List[Optimizer, ...] or Tuple[Optimizer, ...]"
            self._opts = opt_conf
        else:
            assert False, "model.configure_optimizers() must return Optimizer "
            "or List[Optimizer, ...] or Tuple[Optimizer, ...]"

        return True

    def _get_and_check_jobs(self):
        # TOOD(strint): rm numpy in sub-model
        self._is_numpy_input = (
            True if isinstance(self._cfg.data, NumpyDataModule) else False
        )
        self._jobs = []

        if self._is_numpy_input:
            self._first_numpy_batch = []
            for optimizer_idx in range(0, len(self._opts)):
                batch = self._cfg.data(0, optimizer_idx)
                self._first_numpy_batch.insert(optimizer_idx, batch)
                self._jobs.insert(
                    optimizer_idx, self._construct_numpy_job(batch, optimizer_idx)
                )
        else:
            for optimizer_idx in range(0, len(self._opts)):
                self._jobs.insert(optimizer_idx, self._construct_job(optimizer_idx))

        return True

    def _construct_job(self, optimizer_idx: int = 0):
        def job():
            batch = self._cfg.data(0, optimizer_idx)
            outputs = self._model.training_step(
                batch=batch, optimizer_idx=optimizer_idx
            )
            loss = None
            if isinstance(outputs, tuple) and len(outputs) > 0:
                loss = outputs[0]
            else:
                loss = outputs
            self._opts[optimizer_idx].minimize(loss)
            return outputs

        job.__name__ = (
            self._model.__class__.__name__ + "_Model_train_job_" + str(optimizer_idx)
        )
        deco = api_oneflow_function(type="train", function_config=self._cfg.exe_cfg)
        return deco(job)

    def _construct_numpy_job(self, batch, optimizer_idx):
        def job(*input_batch):
            outputs = self._model.training_step(
                batch=input_batch, optimizer_idx=optimizer_idx
            )
            loss = None
            if isinstance(outputs, tuple) and len(outputs) > 0:
                loss = outputs[0]
            else:
                loss = outputs
            self._opts[optimizer_idx].minimize(loss)
            return outputs

        _infer_job_signature(self._cfg.data, batch, optimizer_idx, job)

        job.__name__ = (
            self._model.__class__.__name__
            + "_Model_train_numpy_job_"
            + str(optimizer_idx)
        )
        deco = api_oneflow_function(type="train", function_config=self._cfg.exe_cfg)
        return deco(job)


class ValidateModel(SubModel):
    def __init__(
        self,
        cfg: ValidationConfig = None,
        model: Model = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ):
        super().__init__("validate_model", cfg, model, callbacks)

        if not self._get_and_check_step():
            self.is_valid = False

        if self.is_valid and not self._get_and_check_job():
            self.is_valid = False

    def step(self, step_idx: int = 0):
        assert self.is_valid
        if (step_idx + 1) % self._cfg.step_interval == 0:
            outputs = None
            if self._is_numpy_input:
                batch = None
                if step_idx == 0:
                    batch = self._first_numpy_batch
                else:
                    batch = self._cfg.data(step_idx)
                outputs = self._job(*batch).get()
            else:
                outputs = self._job().get()
            self._method_callback(
                "on_validation_step_end", step_idx=step_idx, outputs=outputs,
            )

    def _get_and_check_step(self):
        if not self._model.method_overrided("validation_step"):
            self.error_msg += "model.validation_step() is empty;"
            return False
        else:
            return True

    def _get_and_check_job(self):
        # TOOD(strint): rm numpy in sub-model
        self._is_numpy_input = (
            True if isinstance(self._cfg.data, NumpyDataModule) else False
        )
        self._job = None
        if not self._is_numpy_input:
            self._job = self._construct_job()
        else:
            batch = self._cfg.data(0)
            self._first_numpy_batch = batch
            self._job = self._construct_numpy_job(batch)

        return True

    def _construct_job(self):
        def job():
            batch = self._cfg.data()
            return self._model.validation_step(batch)

        job.__name__ = self._model.__class__.__name__ + "_Model_eval_job"
        deco = api_oneflow_function(type="predict", function_config=self._cfg.exe_cfg)
        return deco(job)

    def _construct_numpy_job(self, batch: Tuple[np.ndarray, ...] = None):
        def job(*input_batch):
            return self._model.validation_step(batch=input_batch)

        _infer_job_signature(self._cfg.data, batch, 0, job)

        job.__name__ = self._model.__class__.__name__ + "_Model_eval_numpy_job"
        deco = api_oneflow_function(type="predict", function_config=self._cfg.exe_cfg)
        return deco(job)


class CheckpointModel(SubModel):
    def __init__(
        self,
        cfg: CheckpointConfig = None,
        model: Model = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ):
        super().__init__("checkpoint_model", cfg, model, callbacks)

    def load(self):
        assert self.is_valid
        if self._cfg.need_load:
            self._load_checkpoint(self._cfg.load_dirpath)

    def step(self, step_idx: int = 0):
        assert self.is_valid
        if self._cfg.need_save:
            if (step_idx + 1) % self._cfg.save_step_interval == 0:
                self._save_checkpoint(
                    dirpath=self._cfg.save_dirpath + "-" + str(step_idx)
                )

    def _load_checkpoint(
        self, dirpath: str,
    ):
        r"""Load model states from a checkpoint.
        """
        LoadVariables(GetCheckpoint(path=dirpath))

    def _save_checkpoint(
        self, dirpath: str,
    ):
        r"""Save model states as a checkpoint.
        """
        SaveVarDict(path=dirpath)


def _infer_job_signature(data_module, batch, optimizer_idx, job):
    para_list = []
    placeholder_list = data_module.infer_oneflow_data_placeholder(batch, optimizer_idx)
    for i, placeholder in enumerate(placeholder_list):
        para_name = (
            data_module.__class__.__name__
            + "_opt_"
            + str(optimizer_idx)
            + "_para_"
            + str(i)
        )
        para_list.append(
            inspect.Parameter(
                name=para_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=placeholder,
            )
        )

    origin_sig = inspect.signature(job)
    new_sig = origin_sig.replace(parameters=para_list)
    job.__oneflow_function_signature__ = new_sig
