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
__all__ = [
    "DataModule",
    "NumpyDataModule",
    "TrainingConfig",
    "ValidationConfig",
    "CheckpointConfig",
    "Callback",
    "Model",
]
import inspect
from abc import ABC
from typing import Any, List, Optional, Tuple, Union

import numpy as np

import oneflow as flow
import oneflow._oneflow_internal
import oneflow.framework.dtype as dtype_util
from oneflow.framework.function_util import FunctionConfig as ExecutionConfig
from oneflow.framework.tensor import Tensor
from oneflow.nn.modules.module import Module
from oneflow.optim.optimizer import Optimizer as OOPOptimizer


class DataModule(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, step_idx: int = 0, optimizer_idx: int = 0):
        pass

    def infer_oneflow_data_placeholder(
        self, batch: Tuple[Any] = None, optimizer_idx: int = 0
    ):
        return None


class NumpyDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, step_idx: int = 0, optimizer_idx: int = 0):
        pass

    def __call__(self, *args):
        ret = self.forward(*args)
        return ret

    def infer_oneflow_data_placeholder(
        self, batch: Tuple[np.ndarray, ...] = None, optimizer_idx: int = 0
    ):
        assert isinstance(batch, tuple), "model.NumpyDataModule must return a tuple."
        data_placeholder_list = []
        for item in batch:
            assert isinstance(
                item, np.ndarray
            ), "model.NumpyDataModule must return a tuple of numpy."
            of_dtype = dtype_util.convert_numpy_dtype_to_oneflow_dtype(item.dtype)
            # numpy_placeholder = oneflow_typing.Numpy.Placeholder(
            #    shape=item.shape, dtype=of_dtype
            # )
            data_placeholder_list.append(numpy_placeholder)
        return data_placeholder_list


class TrainingConfig:
    def __init__(self):
        super().__init__()
        self.exe_cfg = ExecutionConfig()
        self.data = None
        self.error_msg = ""

    def config_execution(self, exe_cfg: ExecutionConfig = None):
        self.exe_cfg = exe_cfg

    def config_data(self, data: DataModule = None):
        self.data = data

    def check_valid(self):
        is_valid = True
        self.error_msg = ""
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


class ValidationConfig:
    def __init__(self):
        super().__init__()
        self.exe_cfg = ExecutionConfig()
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
        if self.data is None:
            self.error_msg += "model.ValidationConfig data is None;"
            is_valid = False
        if not isinstance(self.data, DataModule):
            self.error_msg += "model.ValidationConfig data is not DataModule;"
            is_valid = False
        if self.step_interval <= 0 or not isinstance(self.step_interval, int):
            self.error_msg += (
                "model.ValidationConfig step_interval is <= 0 or is not int;"
            )
            is_valid = False
        return is_valid


class CheckpointConfig(object):
    def __init__(self):
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
        assert isinstance(step_interval, int), "step_interval should be int"

    def check_valid(self):
        is_valid = True
        self.error_msg = ""
        return is_valid


class Callback(ABC):
    """ Abstract base class used to build new callbacks.
    """

    def on_training_step_end(
        self,
        outputs: Optional[Union[Tensor, Tuple[Tensor, ...]]],
        step_idx: int = 0,
        optimizer_idx: int = 0,
    ):
        pass

    def on_validation_step_end(
        self, outputs: Optional[Union[Tensor, Tuple[Tensor, ...]]], step_idx: int = 0,
    ):
        pass


class Model(ABC, Module):
    """A high level API for model training and validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._is_deprecated_function_style = (
            kwargs["is_deprecated_function_style"]
            if "is_deprecated_function_style" in kwargs
            else False
        )

    def forward(self, *args, **kwargs):
        """Same as `nn.Module.forward()`, here is to define the operations you want to use for prediction.
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        """Operates on a single batch of data from the training set and return loss.
        """
        raise NotImplementedError()

    def validation_step(self, *args, **kwargs):
        """Operates on a single batch of data from the validation set.
        """
        raise NotImplementedError()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
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
        """ Runs the full training and validation routine.
        """
        self._max_steps = max_steps
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
                        "Model step_idx {} {} failed.".format(step_idx, sub_model.name)
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
        self._train_model = (
            TrainModel(training_config, self, callbacks)
            if self._is_deprecated_function_style
            else TrainModelOOPStyle(training_config, self, callbacks)
        )
        if self._train_model.is_valid:
            sub_models.append(self._train_model)
        elif training_config is not None:
            print(
                self._train_model.error_msg,
                "{}'s fit() will not do training.".format(self.__class__.__name__),
            )
        self._val_model = (
            ValidateModel(validation_config, self, callbacks)
            if self._is_deprecated_function_style
            else ValidateModelOOPStyle(validation_config, self, callbacks)
        )
        if self._val_model.is_valid:
            sub_models.append(self._val_model)
        elif validation_config is not None:
            print(
                self._val_model.error_msg,
                "{}'s fit() will not do validation.".format(self.__class__.__name__),
            )
        if len(sub_models) == 0:
            print(
                "{}'s fit() will do nothing because there has no valid configuration.".format(
                    self.__class__.__name__
                )
            )
            return sub_models
        self._checkpoint_model = (
            CheckpointModel(checkpoint_config, self, callbacks)
            if self._is_deprecated_function_style
            else CheckpointModelOOPStyle(checkpoint_config, self, callbacks)
        )
        if self._checkpoint_model.is_valid:
            sub_models.append(self._checkpoint_model)
        elif checkpoint_config is not None:
            print(
                self._checkpoint_model.error_msg,
                "{}'s fit() will not do checkpoint.".format(self.__class__.__name__),
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
        self.error_msg = (
            self._model.__class__.__name__ + " " + self.name + " error message: "
        )
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
        super().__init__("training", cfg, model, callbacks)
        if not self._get_and_check_step():
            self.is_valid = False
        if not self._get_and_check_opts():
            self.is_valid = False
        if self.is_valid and (not self._get_and_check_jobs()):
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
                ), "model.configure_optimizers() must return Optimizer                     or List[Optimizer, ...] or Tuple[Optimizer, ...]"
            self._opts = opt_conf
        else:
            assert (
                False
            ), "model.configure_optimizers() must return Optimizer                 or List[Optimizer, ...] or Tuple[Optimizer, ...]"
        return True

    def _get_and_check_jobs(self):
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
        deco  # = api_oneflow_function(type="train", function_config=self._cfg.exe_cfg)
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
        deco  # = api_oneflow_function(type="train", function_config=self._cfg.exe_cfg)
        return deco(job)


class ValidateModel(SubModel):
    def __init__(
        self,
        cfg: ValidationConfig = None,
        model: Model = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ):
        super().__init__("validation", cfg, model, callbacks)
        if not self._get_and_check_step():
            self.is_valid = False
        if self.is_valid and (not self._get_and_check_job()):
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
                    batch = self._cfg.data(step_idx, 0)
                outputs = self._job(*batch).get()
            else:
                outputs = self._job().get()
            self._method_callback(
                "on_validation_step_end", step_idx=step_idx, outputs=outputs
            )

    def _get_and_check_step(self):
        if not self._model.method_overrided("validation_step"):
            self.error_msg += "model.validation_step() is empty;"
            return False
        else:
            return True

    def _get_and_check_job(self):
        self._is_numpy_input = (
            True if isinstance(self._cfg.data, NumpyDataModule) else False
        )
        self._job = None
        if not self._is_numpy_input:
            self._job = self._construct_job()
        else:
            batch = self._cfg.data(0, 0)
            self._first_numpy_batch = batch
            self._job = self._construct_numpy_job(batch)
        return True

    def _construct_job(self):
        def job():
            batch = self._cfg.data(0, 0)
            return self._model.validation_step(batch)

        job.__name__ = self._model.__class__.__name__ + "_Model_eval_job"
        deco  # = api_oneflow_function(type="predict", function_config=self._cfg.exe_cfg)
        return deco(job)

    def _construct_numpy_job(self, batch: Tuple[np.ndarray, ...] = None):
        def job(*input_batch):
            return self._model.validation_step(batch=input_batch)

        _infer_job_signature(self._cfg.data, batch, 0, job)
        job.__name__ = self._model.__class__.__name__ + "_Model_eval_numpy_job"
        deco  # = api_oneflow_function(type="predict", function_config=self._cfg.exe_cfg)
        return deco(job)


class CheckpointModel(SubModel):
    def __init__(
        self,
        cfg: CheckpointConfig = None,
        model: Model = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ):
        super().__init__("checkpointing", cfg, model, callbacks)

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

    def _load_checkpoint(self, dirpath: str):
        """Load model states from a checkpoint.
        """
        stat_dict = flow.load(path=dirpath)
        self._model.load_state_dict(stat_dict)

    def _save_checkpoint(self, dirpath: str):
        """Save model states as a checkpoint.
        """
        stat_dict = self._model.state_dict()
        flow.save(stat_dict, dirpath)


class TrainModelOOPStyle(SubModel):
    def __init__(
        self,
        cfg: TrainingConfig = None,
        model: Model = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ):
        super().__init__("training", cfg, model, callbacks)
        if not self._get_and_check_step():
            self.is_valid = False
        if not self._get_and_check_opts():
            self.is_valid = False

    def step(self, step_idx: int = 0):
        assert self.is_valid, self.error_msg
        for optimizer_idx in range(0, len(self._opts)):
            batch = self._cfg.data(step_idx, optimizer_idx)
            outputs = self._model.training_step(
                batch=batch, optimizer_idx=optimizer_idx
            )
            loss = None
            if isinstance(outputs, tuple) and len(outputs) > 0:
                loss = outputs[0]
            else:
                loss = outputs
            loss.backward()
            opt = self._opts[optimizer_idx]
            opt.step()
            opt.zero_grad()
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
        if isinstance(opt_conf, OOPOptimizer):
            self._opts = [opt_conf]
        elif isinstance(opt_conf, (list, tuple)):
            for opt in opt_conf:
                assert isinstance(
                    opt, OOPOptimizer
                ), "model.configure_optimizers() must return Optimizer                     or List[Optimizer, ...] or Tuple[Optimizer, ...]"
            self._opts = opt_conf
        else:
            assert (
                False
            ), "model.configure_optimizers() must return Optimizer                 or List[Optimizer, ...] or Tuple[Optimizer, ...]"
        return True


class ValidateModelOOPStyle(SubModel):
    def __init__(
        self,
        cfg: ValidationConfig = None,
        model: Model = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ):
        super().__init__("validation", cfg, model, callbacks)
        if not self._get_and_check_step():
            self.is_valid = False

    def step(self, step_idx: int = 0):
        assert self.is_valid
        if (step_idx + 1) % self._cfg.step_interval == 0:
            outputs = None
            with oneflow._oneflow_internal.autograd.no_grad():
                inputs = self._cfg.data(step_idx, 0)
                model_previous_mode = self._model.training
                self._model.train()
                outputs = self._model.validation_step(inputs)
                self._model.train(model_previous_mode)
            self._method_callback(
                "on_validation_step_end", step_idx=step_idx, outputs=outputs
            )

    def _get_and_check_step(self):
        if not self._model.method_overrided("validation_step"):
            self.error_msg += "model.validation_step() is empty;"
            return False
        else:
            return True


class CheckpointModelOOPStyle(SubModel):
    def __init__(
        self,
        cfg: CheckpointConfig = None,
        model: Model = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ):
        super().__init__("checkpointing", cfg, model, callbacks)

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

    def _load_checkpoint(self, dirpath: str):
        """Load model states from a checkpoint.
        """
        stat_dict = flow.load(path=dirpath)
        self._model.load_state_dict(stat_dict)

    def _save_checkpoint(self, dirpath: str):
        """Save model states as a checkpoint.
        """
        stat_dict = self._model.state_dict()
        flow.save(stat_dict, dirpath)


def _infer_job_signature(data_module, batch, optimizer_idx, job):
    para_list = []
    placeholder_list = data_module.infer_oneflow_data_placeholder(batch, optimizer_idx)
    for (i, placeholder) in enumerate(placeholder_list):
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
