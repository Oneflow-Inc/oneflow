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
from collections import OrderedDict
from functools import partial
from typing import Iterator, Optional, Set, Union

import oneflow._oneflow_internal
import oneflow.framework.graph_build_util as graph_build_util
from oneflow.framework.distribute import get_rank
from oneflow.framework.tensor import Tensor, TensorTuple
from oneflow.nn.module import Module
from oneflow.nn.parameter import Parameter
from oneflow.nn.graph.util import add_indent


class BlockType:
    NONE = "NONE"
    MODULE = "MODULE"
    PARAMETER = "PARAMETER"
    BUFFER = "BUFFER"


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
        self._debug = False
        if isinstance(value, Module):
            self._type = BlockType.MODULE
            self._is_executing_forward = False
            self._modules = OrderedDict()
            self._parameters = OrderedDict()
            self._buffers = OrderedDict()
            for (n, m) in list(value.named_children()):
                self.__setattr__(n, Block(self._name_prefix + self._name + ".", n, m))
            for (n, p) in list(value.named_parameters("", False)):
                self.__setattr__(n, Block(self._name_prefix + self._name + ".", n, p))
            for (n, b) in list(value.named_buffers("", False)):
                self.__setattr__(n, Block(self._name_prefix + self._name + ".", n, b))
            self._args_repr = []
            self._outs_repr = []
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

    def debug(self, mode: bool = True) -> None:
        if get_rank() != 0:
            return
        self._debug = mode
        if self._type == BlockType.MODULE:

            def _set_child(d):
                for (_, n) in d.items():
                    n.debug(mode)

            _set_child(self._modules)
            _set_child(self._parameters)
            _set_child(self._buffers)

    def scope_context(self):
        return graph_build_util.BlockScopeContext(self.prev_scope, self.scope)

    def __call__(self, *args):
        assert self._type == BlockType.MODULE
        if self._debug:
            print(self._shallow_repr())

        for idx, arg in enumerate(args):
            meta_repr_str = (
                arg._meta_repr() if isinstance(arg, Tensor) else str(type(arg))
            )
            in_str = (
                "(INPUT:_"
                + self.name_prefix
                + self.name
                + "-input_"
                + str(idx)
                + ":"
                + meta_repr_str
                + ")"
            )
            if not isinstance(arg, Tensor):
                in_str = "[WARNING]" + in_str
            self._args_repr.append(in_str)
            if self._debug:
                print(in_str)

                def _print_state(d):
                    for (_, n) in d.items():
                        print(n._shallow_repr())

                _print_state(self._parameters)
                _print_state(self._buffers)

        result = self._origin.__class__.__call__(self, *args)

        outputs = ()
        if not (type(result) is tuple or type(result) is list):
            outputs = (result,)
        else:
            outputs = result

        for idx, out in enumerate(outputs):
            out_repr = out._meta_repr() if isinstance(out, Tensor) else str(type(out))
            out_str = (
                "(OUTPUT:_"
                + self.name_prefix
                + self.name
                + "-output_"
                + str(idx)
                + ":"
                + out_repr
                + ")"
            )
            if not isinstance(out, Tensor):
                out_str = "[WARNING]" + out_str

            self._outs_repr.append(out_str)
            if self._debug:
                print(out_str)

        return result

    def __iter__(self) -> Iterator["Block"]:
        assert self._type == BlockType.MODULE
        return iter(self._modules.values())

    def forward(self, *args):
        assert self._type == BlockType.MODULE
        self._is_executing_forward = True
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
            for (name, module) in self._modules.items():
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
            for (k, v) in members:
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
            # support get module
            if "_modules" in self.__dict__:
                modules = self.__dict__["_modules"]
                if name in modules:
                    return modules[name]
            # support get parameter
            p_state = self._get_in_states(name, "_parameters")
            if p_state is not None:
                return p_state
            # support get buffer
            b_state = self._get_in_states(name, "_buffers")
            if b_state is not None:
                return b_state
            # support get normal attr
            if name in self._origin.__dict__:
                return self._origin.__dict__[name]
            # support get function
            if hasattr(self._origin, name):
                return partial(getattr(self._origin.__class__, name), self)
        raise AttributeError(
            "'{}' '{}' object '{}' in nn.Graph has no attribute '{}'".format(
                self._type, type(self).__name__, self._name_prefix + self.name, name
            )
        )

    def _get_in_states(self, name, states_name):
        if states_name not in self.__dict__:
            return None

        _states = self.__dict__[states_name]
        if name not in _states:
            return None

        _s_block = _states[name]
        if graph_build_util.lazy_mode.is_enabled():
            #  lazy
            if _s_block._lazy_origin is None:
                assert _s_block._lazy_origin_builder is not None, (
                    repr(_s_block) + " has no lazy Tensor creation function."
                )
                assert self._is_executing_forward, (
                    repr(_s_block)
                    + "'s first get must happened in it's nn.Module.forward() to generate the right scope."
                )
                with _s_block.scope_context():
                    _s_block._lazy_origin = _s_block._lazy_origin_builder()
            return _s_block._lazy_origin
        elif (
            not graph_build_util.lazy_mode.is_enabled()
        ) and self._is_executing_forward:
            # eager and inside nn.Graph.build()
            return _s_block.origin
        else:
            # outside nn.Graph.build()
            return _s_block

    def __repr__(self):
        lines = None
        if self._type == BlockType.MODULE:
            child_lines = []
            if len(self._args_repr) > 0:
                for in_str in self._args_repr:
                    input_str = add_indent(in_str, 2)
                    child_lines.append(input_str)

            def _append_child(d):
                for (_, n) in d.items():
                    n_str = repr(n)
                    n_str = add_indent(n_str, 2)
                    child_lines.append(n_str)

            _append_child(self._parameters)
            _append_child(self._buffers)
            _append_child(self._modules)

            if len(self._outs_repr) > 0:
                for out_str in self._outs_repr:
                    output_str = add_indent(out_str, 2)
                    child_lines.append(output_str)

            if len(child_lines) > 0:
                lines = child_lines
        main_str = self._shallow_repr() + ": ("
        if lines is not None:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def _shallow_repr(self):
        shallow_repr = (
            "("
            + self._type
            + ":"
            + self._name_prefix
            + self._name
            + ":"
            + (
                self._origin._shallow_repr()
                if self._type == BlockType.MODULE
                else (self._origin._meta_repr())
            )
            + ")"
        )
        return shallow_repr


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
