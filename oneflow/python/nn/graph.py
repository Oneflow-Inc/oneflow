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
from typing import Union

import oneflow._oneflow_internal
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.graph_build_util as graph_build_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.tensor_tuple_util as tensor_tuple_util
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.multi_client_session import MultiClientSession
from oneflow.python.framework.tensor import Tensor
from oneflow.python.nn.module import Module
from oneflow.python.nn.parameter import Parameter
from oneflow.python.nn.optimizer.optimizer import Optimizer
from oneflow.python.framework.function_util import FunctionConfig


@oneflow_export("nn.Graph", "nn.graph.Graph")
@experimental_api
class Graph(object):
    _child_init_cnt = dict()

    def __init__(self):
        self.config = GraphConfig()
        self._generate_name()
        self.config.proto.set_job_name(self._name)
        self._c_nn_graph = oneflow._oneflow_internal.NNGraph(self._name)
        self._blocks = OrderedDict()
        self._optimizers = OrderedDict()
        self._is_compiled = False
        self._state_tensortuple = None

    @property
    def name(self):
        return self._name

    @property
    def training(self):
        return self.config.training

    @property
    def _graph_proto(self):
        return c_api_util.GetCurrentJob()

    def build(self, *args):
        raise NotImplementedError()

    def add_optimizer(
        self,
        name: str,
        optimizer: Optimizer = None,
        lr_scheduler=None,
        grad_clipping_conf=None,
        weight_decay_conf=None,
    ):
        self._optimizers[name] = self.OptimizerConfig(
            optimizer, lr_scheduler, grad_clipping_conf, weight_decay_conf
        )

    def _generate_name(self):
        child_name = self.__class__.__name__
        if Graph._child_init_cnt.get(child_name) is None:
            Graph._child_init_cnt[child_name] = 0
        self._name = child_name + "_" + str(Graph._child_init_cnt[child_name])
        Graph._child_init_cnt[child_name] += 1

    def _named_state(self):
        for _, b in self._blocks.items():
            prefix = b.name + "."
            p_gen = b.origin.named_parameters()
            for n, p in p_gen:
                yield prefix + n, p
            b_gen = b.origin.named_buffers()
            for n, b in b_gen:
                yield prefix + n, b

    def _compile(self, *args):
        assert not self._is_compiled, (
            "nn.Graph " + self._name + " has already been compiled."
        )
        state = tuple(t for _, t in self._named_state())
        if len(state) > 0:
            self._state_tensortuple = tensor_tuple_util.convert_to_tensor_tuple(state)

        session = session_ctx.GetDefaultSession()
        assert type(session) is MultiClientSession
        session.TryInit()

        with graph_build_util.graph_build_context(self.config.proto, session):
            outputs = self.build(*args)

        self._is_compiled = True

    def _launch(self):
        # TODO(xuxiaoyu)
        # return self._c_nn_graph.run()
        ...

    def __call__(self, *args):
        # TODO(xuxiaoyu)
        # if not self._is_compiled:
        #     self._compile()
        # return self._launch()
        ...

    def _add_block(self, name: str, module: Module = None) -> None:
        r"""Adds a module to the current graph as a block.

        The block can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child block. The child block can be
                accessed from this graph using the given name
            module (Module): child module to be added to the graph.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(type(name)))
        elif hasattr(self, name) and name not in self._blocks:
            raise KeyError("attribute '{}' already exists".format(name))
        elif "." in name:
            raise KeyError('module name can\'t contain ".", got: {}'.format(name))
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        self._blocks[name] = Block("", name, module)

    def __setattr__(self, name: str, value=None):
        if isinstance(value, Module):
            self._add_block(name, value)
        elif isinstance(value, Optimizer):
            raise AttributeError(
                "'{}' object are not allowed to set Optimizer attribute named '{}', \
                 please use add_optimizer(...) instead.".format(
                    type(self).__name__, name
                )
            )
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        if "_blocks" in self.__dict__:
            if name in self._blocks:
                return self._blocks[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

    def __repr__(self):
        lines = None
        if len(self._blocks) > 0:
            child_lines = []
            for n, m in self._blocks.items():
                mod_str = repr(m)
                mod_str = _add_indent(mod_str, 2)
                child_lines.append(mod_str)
            lines = child_lines

        main_str = "(" + self._name + ":" + self.__class__.__name__ + ":GRAPH): ("
        if lines is not None:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


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
        self._config = BlockConfig()

        if isinstance(value, Module):
            self._type = BlockType.MODULE
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
        elif isinstance(value, Tensor):
            self._type = BlockType.BUFFER
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

    def __call__(self, *args):
        assert self._type == BlockType.MODULE
        # TODO(): with oneflow_c_api.set_scope(self.config_):
        return self._origin.__class__.__call__(self, *args)

    def forward(self, *args):
        assert self._type == BlockType.MODULE
        return self._origin.__class__.forward(self, *args)

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
                    # TODO(): return block when need config
                    # return _parameters[name]
                    return _parameters[name].origin
            if "_buffers" in self.__dict__:
                _buffers = self.__dict__["_buffers"]
                if name in _buffers:
                    # TODO(): return block when need config
                    # return _buffers[name]
                    return _buffers[name].origin
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
                    n_str = _add_indent(n_str, 2)
                    child_lines.append(n_str)

            _append_child(self._modules)
            _append_child(self._parameters)
            _append_child(self._buffers)
            if len(child_lines) > 0:
                lines = child_lines

        main_str = (
            "("
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

    @property
    def scope(self):
        return self._config.scope


@oneflow_export("nn.graph.GraphConfig")
@experimental_api
class GraphConfig(FunctionConfig):
    def __init__(self):
        super().__init__()
        self._train(False)

    @property
    def proto(self):
        return self.function_desc.job_config_proto

    @property
    def training(self):
        if self.function_desc.job_config_proto.has_train_conf():
            return True
        if self.function_desc.job_config_proto.has_predict_conf():
            return False
        raise NotImplementedError

    def _train(self, mode: bool = True):
        if mode:
            self.function_desc.job_config_proto.mutable_train_conf()
        else:
            self.function_desc.job_config_proto.mutable_predict_conf()


@oneflow_export("nn.graph.BlockConfig")
@experimental_api
class BlockConfig(object):
    def __init__(self):
        self._stage_id = None
        self._activation_checkpointing = False

    @property
    def scope(self):
        # TODO(xuxiaoyu): support generating Scope Object
        print("BlockConfig.scope todo")

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


def _add_indent(in_s, num_spaces):
    s = in_s.split("\n")
    if len(s) == 1:
        return in_s
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s
