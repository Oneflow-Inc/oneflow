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
from collections import OrderedDict
from typing import Union, Optional, Iterator, Set

import oneflow._oneflow_internal
import oneflow.python.framework.graph_build_util as graph_build_util
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import Tensor
from oneflow.python.nn.module import Module
from oneflow.python.nn.parameter import Parameter
from oneflow.python.nn.util import add_indent


class BlockType:
    NONE = "NONE"
    MODULE = "MODULE"
    PARAMETER = "PARAMETER"
    BUFFER = "BUFFER"


@oneflow_export("nn.graph.Block")
@experimental_api
class Block(object):
    def __init__(
        self,
        prefix: str = "",
        name: str = "",
        value: Union[Module, Parameter, Tensor] = None,
    ):
        assert not isinstance(value, Block)
        self._name = name
        self._name_prefix = prefix
        self._type = BlockType.NONE
        self._origin = value
        self.config = BlockConfig()
        self._scope = None
        self._prev_scope = None

        if isinstance(value, Module):
            self._type = BlockType.MODULE
            self._is_executing_forward = False
            self._modules = OrderedDict()
            self._parameters = OrderedDict()
            self._buffers = OrderedDict()
            for n, m in list(value.named_children()):
                self.__setattr__(n, Block(self._name_prefix + self._name + ".", n, m))
            for n, p in list(value.named_parameters("", False)):
                self.__setattr__(n, Block(self._name_prefix + self._name + ".", n, p))
            for n, b in list(value.named_buffers("", False)):
                self.__setattr__(n, Block(self._name_prefix + self._name + ".", n, b))
        elif isinstance(value, Parameter):
            self._type = BlockType.PARAMETER
            self._lazy_origin = None
            self._lazy_origin_builder = None
        elif isinstance(value, Tensor):
            self._type = BlockType.BUFFER
            self._lazy_origin = None
            self._lazy_origin_builder = None
        else:
            raise NotImplementedError()

    @property
    def name(self):
        return self._name

    @property
    def name_prefix(self):
        return self._name_prefix

    @property
    def type(self):
        return self._type

    @property
    def origin(self):
        return self._origin

    @property
    def lazy_origin(self):
        assert (
            self._type == BlockType.PARAMETER or self._type == BlockType.BUFFER
        ), "Only Parameter or Buffer Block has lazy_origin"
        return self._lazy_origin

    def lazy_origin_builder(self):
        assert (
            self._type == BlockType.PARAMETER or self._type == BlockType.BUFFER
        ), "Only Parameter or Buffer Block has lazy_origin_builder"
        return self._lazy_origin_builder

    def set_lazy_origin_builder(self, fn=None):
        assert (
            self._type == BlockType.PARAMETER or self._type == BlockType.BUFFER
        ), "Only Parameter or Buffer Block has lazy_origin_builder"
        self._lazy_origin_builder = fn

    @property
    def prev_scope(self):
        if self._prev_scope is None:
            self._prev_scope = oneflow._oneflow_internal.GetCurrentScope()
        return self._prev_scope

    @property
    def scope(self):
        if self._scope is None:
            self._scope = graph_build_util.make_new_block_scope(self.prev_scope, self)
        return self._scope

    def scope_context(self):
        return graph_build_util.BlockScopeContext(self.prev_scope, self.scope)

    def __call__(self, *args):
        assert self._type == BlockType.MODULE
        # nn.Module.__call__ will call self.forward()
        # so the scope is set in self.forward()
        result = self._origin.__class__.__call__(self, *args)
        return result

    def forward(self, *args):
        assert self._type == BlockType.MODULE
        self._is_executing_forward = True
        # TODO(xuxiaoyu): only build scope in lazy mode
        with self.scope_context():
            result = self._origin.__class__.forward(self, *args)
        self._is_executing_forward = False
        return result

    def modules(self, memo: Optional[Set["Block"]] = None) -> Iterator["Block"]:
        assert self._type == BlockType.MODULE
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield self
            for name, module in self._modules.items():
                if module is None:
                    continue
                for m in module.modules(memo):
                    yield m

    def _members(self, get_members_fn, recurse=True) -> Iterator["Block"]:
        assert self._type == BlockType.MODULE
        memo = set()
        modules = self.modules() if recurse else [self]
        for module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                yield v

    def parameters(self, recurse: bool = True) -> Iterator["Block"]:
        assert self._type == BlockType.MODULE
        gen = self._members(lambda module: module._parameters.items(), recurse=recurse)
        for elem in gen:
            yield elem

    def buffers(self, recurse: bool = True) -> Iterator["Block"]:
        assert self._type == BlockType.MODULE
        gen = self._members(lambda module: module._buffers.items(), recurse=recurse)
        for elem in gen:
            yield elem

    def __setattr__(self, name: str, value=None) -> None:
        if value is None or not isinstance(value, Block):
            self.__dict__[name] = value
        else:
            dicts_or_sets = (
                self.__dict__,
                self._modules,
                self._parameters,
                self._buffers,
            )
            for d in dicts_or_sets:
                if name in d:
                    raise AttributeError(
                        "'{}' object has duplicated attribute named '{}'".format(
                            self._name, name
                        )
                    )
            if value.type == BlockType.MODULE:
                self._modules[name] = value
            elif value.type == BlockType.PARAMETER:
                self._parameters[name] = value
            elif value.type == BlockType.BUFFER:
                self._buffers[name] = value
            else:
                raise AttributeError(
                    "'{}' object are not allowed to set attribute named '{}'".format(
                        type(self).__name__, name
                    )
                )

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return self.__dict__[name]

        if self._type == BlockType.MODULE:
            if "_modules" in self.__dict__:
                modules = self.__dict__["_modules"]
                if name in modules:
                    return modules[name]
            if "_parameters" in self.__dict__:
                _parameters = self.__dict__["_parameters"]
                if name in _parameters:
                    p_block = _parameters[name]
                    if self._is_executing_forward:
                        # Return Tensor for running when getattr is
                        # inside it's father Block's forward()
                        if graph_build_util.lazy_mode.is_enabled():
                            if p_block._lazy_origin is None:
                                assert p_block._lazy_origin_builder is not None, (
                                    repr(p_block)
                                    + " has no lazy Tensor creation function."
                                )
                                # Create and return lazy tensor
                                with p_block.scope_context():
                                    p_block._lazy_origin = (
                                        p_block._lazy_origin_builder()
                                    )
                            return p_block._lazy_origin
                        else:
                            return p_block.origin
                    else:
                        # Return Block for config when getattr is
                        # outside it's father Block's forward()
                        return p_block
            if "_buffers" in self.__dict__:
                _buffers = self.__dict__["_buffers"]
                if name in _buffers:
                    b_block = _buffers[name]
                    if self._is_executing_forward:
                        # Return Tensor for running when getattr is
                        # inside it's father Block's forward()
                        if graph_build_util.lazy_mode.is_enabled():
                            if b_block._lazy_origin is None:
                                assert b_block._lazy_origin_builder is not None, (
                                    repr(b_block)
                                    + " has no lazy Tensor creation function."
                                )
                                # Create and return lazy tensor
                                with b_block.scope_context():
                                    b_block._lazy_origin = (
                                        b_block._lazy_origin_builder()
                                    )
                            return b_block._lazy_origin
                        else:
                            return b_block.origin
                    else:
                        # Return Block for config when getattr is
                        # outside it's father Block's forward()
                        return b_block
            if name in self._origin.__dict__:
                return self._origin.__dict__[name]

        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

    def __repr__(self):
        lines = None
        if self._type == BlockType.MODULE:
            child_lines = []

            def _append_child(d):
                for _, n in d.items():
                    n_str = repr(n)
                    n_str = add_indent(n_str, 2)
                    child_lines.append(n_str)

            _append_child(self._modules)
            _append_child(self._parameters)
            _append_child(self._buffers)
            if len(child_lines) > 0:
                lines = child_lines

        main_str = (
            "("
            + self._name_prefix
            + self._name
            + ":"
            + self._origin.__class__.__name__
            + ":"
            + self._type
            + "): ("
        )
        if lines is not None:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


@oneflow_export("nn.graph.BlockConfig")
@experimental_api
class BlockConfig(object):
    def __init__(self):
        self._stage_id = None
        self._activation_checkpointing = None

    @property
    def stage_id(self):
        return self._stage_id

    @stage_id.setter
    def stage_id(self, value: int = None):
        self._stage_id = value

    @property
    def activation_checkpointing(self):
        return self._activation_checkpointing

    @activation_checkpointing.setter
    def activation_checkpointing(self, value: bool = False):
        self._activation_checkpointing = value
