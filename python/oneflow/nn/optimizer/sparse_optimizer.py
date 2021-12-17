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


class SparseOptimizer(object):
    r"""SparseOptimizer do not support eager mode for now. If we need sparse optimizer
    in graph mode, use SparseOptimizer to wrap the instance of Optimizer and add SparseOptimizer
    to graph through nn.Graph.add_optimizer.
    """

    def __init__(self, optimizer: Optimizer):
        self._nested_optim = optimizer

    def load_state_dict(self, state_dict):
        self._nested_optim.load_state_dict(state_dict)

    def state_dict(self):
        return self._nested_optim.state_dict()

    def step(self, closure):
        raise NotImplementedError("SparseOptimizer doesn't support step for now")

    def clip_grad(self):
        raise NotImplementedError("SparseOptimizer doesn't support clip_grad for now")

    def zero_grad(self, set_to_none: bool = False):
        raise NotImplementedError("SparseOptimizer doesn't support zero_grad for now")
