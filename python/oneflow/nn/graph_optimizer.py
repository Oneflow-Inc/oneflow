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
from oneflow.nn.optimizer.optimizer import Optimizer
from oneflow.nn.optimizer.lr_scheduler import LrScheduler

class LrWarmUpDict(object):
    def __init__(self, config_dict,):
        assert isinstance(config_dict, dict)
        assert "type" in config_dict
        self._dict = config_dict
        assert self._dict["type"] in ["constent", "linear"]
    
    def generate_conf_for_graph(self, opt_confs):
        if self._dict["type"] == "constent":
            for opt_conf in opt_confs:
                warmup_conf = opt_conf.mutable_warmup_conf()
                warmup_conf.mutable_constant_conf().set_warmup_batches(self._dict["steps"])
                warmup_conf.mutable_constant_conf().set_multiplier(self._dict["multiplier"])
        if self._dict["type"] == "linear":
            for opt_conf in opt_confs:
                warmup_conf = opt_conf.mutable_warmup_conf()
                warmup_conf.mutable_linear_conf().set_warmup_batches(self._dict["steps"])
                warmup_conf.mutable_linear_conf().set_start_multiplier(self._dict["start_multiplier"])


class OptDict(object):
    def __init__(
        self, opt_dict,
    ):
        assert isinstance(opt_dict, dict), "opt dict must be a dict"
        assert "optim" in opt_dict, "opt dict must has an optimizer"
        self._optimizer = opt_dict["optim"]
        assert isinstance(opt_dict["optim"], Optimizer)

        self._lr_scheduler = None
        if "lr_sch" in opt_dict:
            assert isinstance(opt_dict["lr_sch"], LrScheduler)
            self._lr_scheduler = opt_dict["lr_sch"]
            assert (
                self._lr_scheduler._optimizer is self._optimizer
            ), "lr_scheduler's optimizer must be the same optimizer in the opt dict."
        self._lr_warmup = None
        if "lr_warmup" in opt_dict:
            self._lr_warmup = LrWarmUpDict(opt_dict["lr_warmup"])

    def generate_optimizer_and_variable_configs(self, train_conf, vars_conf):
        if self._optimizer is not None:
            opt_confs = self._optimizer.generate_conf_for_graph(train_conf, vars_conf)
        if self._lr_scheduler is not None:
            self._lr_scheduler.generate_conf_for_graph(opt_confs)
        if self._lr_warmup is not None:
            self._lr_warmup.generate_conf_for_graph(opt_confs)


class VariableConfig(object):
    def __init__(self, name: str):
        assert name != ""
        self._name = name
        self._l2 = 0.0

    @property
    def name(self):
        return self._name

    @property
    def l2(self):
        return self._l2

    @l2.setter
    def l2(self, l2: float = 0.0):
        self._l2 = l2

    def __repr__(self):
        return "(variable name: " + self._name + "):(l2: " + str(self._l2) + ".)"
