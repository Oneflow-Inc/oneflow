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
import inspect
import functools

from oneflow.nn.utils.container import *
from oneflow.framework.tensor import Tensor
from oneflow.nn.graph.block import Block, BlockType


# Support oneflow.nn.modules.container in nn.graph as block
# Changeing self._modules or self._parameters to block, when member function called


class SequentialBlock(SequentialContainer[Block]):
    def __init__(self, name: str, super_member: Sequential) -> None:
        self.container_name = name
        super().__init__(super_member)

    def to_block(self):
        if hasattr(self, "_modules"):
            for key, value in self._modules.items():
                if not isinstance(value, Block):
                    self._modules[key] = Block("", self.container_name + key, value)


class BlockModuleList(ModuleList):
    def __init__(self, name: str, super_member: ModuleList) -> None:
        self.container_name = name
        super().__init__(super_member)

    def to_block(self):
        if hasattr(self, "_modules"):
            for key, value in self._modules.items():
                if not isinstance(value, Block):
                    self._modules[key] = Block("", self.container_name + key, value)


class BlockModuleDict(ModuleDict):
    def __init__(self, name: str, super_member: ModuleDict) -> None:
        self.container_name = name
        super().__init__(super_member)

    def to_block(self):
        if hasattr(self, "_modules"):
            for key, value in self._modules.items():
                if not isinstance(value, Block):
                    self._modules[key] = Block("", self.container_name + key, value)


class BlockParameterList(ParameterList):
    def __init__(self, name: str, super_member: ParameterList) -> None:
        self.container_name = name
        super().__init__(super_member)

    def to_block(self):
        if hasattr(self, "_parameters"):
            for key, value in self._parameters.items():
                if not isinstance(value, Block):
                    self._parameters[key] = Block("", self.container_name + key, value)


class BlockParameterDict(ParameterDict):
    def __init__(self, name: str, super_member: ParameterDict) -> None:
        self.container_name = name
        super().__init__(super_member)

    def to_block(self):
        if hasattr(self, "_parameters"):
            for key, value in self._parameters.items():
                if not isinstance(value, Block):
                    self._parameters[key] = Block("", self.container_name + key, value)


def to_block_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        args[0].to_block()
        return result

    return wrapper


block_container = [
    BlockSequential,
    BlockModuleList,
    BlockModuleDict,
    BlockParameterList,
    BlockParameterDict,
]

for container in block_container:
    for name, fn in inspect.getmembers(container, inspect.isfunction):
        if name == "to_block":
            continue
        else:
            setattr(container, name, to_block_decorator(fn))
