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


class OptGroup(object):
    def __init__(
        self, opt_group,
    ):
        assert isinstance(opt_group, dict), "opt group must be a dict"
        assert "optim" in opt_group, "opt group must has an optimizer"
        self._optimizer = opt_group["optim"]
        assert isinstance(opt_group["optim"], Optimizer)
        if "lr_sch" in opt_group:
            assert isinstance(opt_group["lr_sch"], LrScheduler)
            self._lr_scheduler = opt_group["lr_sch"]

    def generate_optimizer_and_variable_configs(self, train_conf, vars_conf):
        if self._optimizer is not None:
            opt_confs = self._optimizer.generate_conf_for_graph(train_conf, vars_conf)
        if self._lr_scheduler is not None:
            self._lr_scheduler.generate_conf_for_graph(opt_confs)


class OptimizerConfig(object):
    def __init__(
        self,
        name: str,
        optimizer: Optimizer = None,
        lr_scheduler: LrScheduler = None,
        # TODO(): support grad clipping
        # grad_clipping_conf=None,
    ):
        self._name = name
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        # self.grad_clipping_conf = grad_clipping_conf

    def generate_optimizer_and_variable_configs(self, train_conf, vars_conf):
        if self._optimizer is not None:
            opt_confs = self._optimizer.generate_conf_for_graph(train_conf, vars_conf)
        if self._lr_scheduler is not None:
            self._lr_scheduler.generate_conf_for_graph(opt_confs)


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
