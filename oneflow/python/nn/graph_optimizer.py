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

from oneflow.python.nn.optimizer.optimizer import Optimizer
from oneflow.python.oneflow_export import oneflow_export, experimental_api


@oneflow_export("nn.graph.OptimizerConfig")
@experimental_api
class OptimizerConfig(object):
    def __init__(
        self,
        name: str,
        optimizer: Optimizer = None,
        lr_scheduler=None,
        grad_clipping_conf=None,
        weight_decay_conf=None,
    ):
        self.name = name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_clipping_conf = grad_clipping_conf
        self.weight_decay_conf = weight_decay_conf
