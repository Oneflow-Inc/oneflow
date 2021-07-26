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
from typing import Dict
from functools import partial

import oneflow._oneflow_internal
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.graph_build_util as graph_build_util
import oneflow.python.framework.session_context as session_ctx
from oneflow._oneflow_internal import Tensor as InternalTensor
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.multi_client_session import MultiClientSession
from oneflow.python.nn.graph_block import Block, BlockType
from oneflow.python.nn.graph_optimizer import OptimizerConfig
from oneflow.python.nn.module import Module
from oneflow.python.nn.optimizer.optimizer import Optimizer
from oneflow.python.nn.util import add_indent
from oneflow.python.framework.function_util import FunctionConfig


@oneflow_export("nn.Graph", "nn.graph.Graph")
@experimental_api
class Graph(object):
    _child_init_cnt = dict()

    def __init__(self):
        self.config = GraphConfig()
        self._generate_name()
        self.config.proto.set_job_name(self._name)
        self._c_nn_graph = oneflow._oneflow_internal.nn.graph.CNNGraph(self._name)
        self._blocks = OrderedDict()
        self._optimizers = OrderedDict()
        self._is_compiled = False
        self._var2var_op_name = dict()
        self._job_proto = None

    @property
    def name(self):
        return self._name

    @property
    def training(self):
        return self.config.training

    @property
    def _graph_proto(self):
        return self._job_proto

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
        assert name is not None, "name cannot be None"
        assert type(name) is str, "name must be an instance of str"
        assert optimizer is not None, "optimizer cannot be None"
        assert isinstance(
            optimizer, Optimizer
        ), "optimizer must be an instance of Optimizer"
        self._optimizers[name] = OptimizerConfig(
            name, optimizer, lr_scheduler, grad_clipping_conf, weight_decay_conf
        )

    def _generate_name(self):
        child_name = self.__class__.__name__
        if Graph._child_init_cnt.get(child_name) is None:
            Graph._child_init_cnt[child_name] = 0
        self._name = child_name + "_" + str(Graph._child_init_cnt[child_name])
        Graph._child_init_cnt[child_name] += 1

    def _state(self):
        for _, b in self._blocks.items():
            pa_gen = b.parameters(recurse=True)
            for pa in pa_gen:
                yield pa
            bu_gen = b.buffers(recurse=True)
            for bu in bu_gen:
                yield bu

    def _preprocess_state(self):
        state_list = list()
        for state_block in self._state():
            state_list.append(state_block.origin)
            if state_block.type == BlockType.PARAMETER:
                self._var2var_op_name[state_block.origin] = (
                    state_block.name_prefix + state_block.name
                )

    def _complete_graph_config(self):
        if len(self._optimizers):
            self.config._train(True)
        # TODO(xuxiaoyu): save variable name and it's l2 if optimizer has weight decay
        # which means to used as l2.
        for name, opt_config in self._optimizers.items():
            self.config.add_optimizer_config(opt_config, self._var2var_op_name)

    def _compile(self, *args):
        assert not self._is_compiled, (
            "nn.Graph " + self._name + " has already been compiled."
        )

        self._preprocess_state()
        self._complete_graph_config()

        session = session_ctx.GetDefaultSession()
        assert type(session) is MultiClientSession
        session.TryInit()
        with graph_build_util.graph_build_context(self.config.proto, session):
            # Deal with input
            lazy_args = []
            lazy_arg_op_names = []
            for idx, arg in enumerate(args):
                op_name = "_" + self.name + "-input_" + str(idx)
                lazy_args.append(graph_build_util.build_graph_input_arg(op_name, arg))
                lazy_arg_op_names.append(op_name)

            # Deal with parameter and buffer
            state_op_names = []
            state_tensors = []
            for state_block in self._state():
                op_name = state_block.name_prefix + state_block.name
                state_tensor = state_block.origin
                state_op_names.append(op_name)
                state_tensors.append(state_tensor)
                state_block.set_lazy_origin_builder(
                    partial(graph_build_util.build_graph_state, op_name, state_tensor)
                )

            # Deal with module in self.build(*args)
            outputs = self.build(*lazy_args)

            # Deal with outputs
            if not (type(outputs) is tuple or type(outputs) is list):
                if outputs is None:
                    outputs = ()
                else:
                    assert type(outputs) is InternalTensor
                    outputs = (outputs,)
            eager_outputs = []
            eager_output_op_names = []
            for idx, out in enumerate(outputs):
                op_name = "_" + self.name + "-output_" + str(idx)
                eager_outputs.append(graph_build_util.build_graph_output(op_name, out))
                eager_output_op_names.append(op_name)
            if len(eager_outputs) == 0:
                eager_outputs = None
            elif len(eager_outputs) == 1:
                eager_outputs = eager_outputs[0]
            else:
                eager_outputs = tuple(eager_outputs)

            # TODO(): call self._c_nn_graph
            #     register lazy_arg_op_names/state_op_names/state_tensors/eager_output_op_names

            # Save job proto for debug
            self._job_proto = c_api_util.GetCurrentJob()

        self._is_compiled = True
        return eager_outputs

    def _launch(self):
        # TODO(xuxiaoyu)
        # return self._c_nn_graph.run()
        ...

    def __call__(self, *args):
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
        # TODO(xuxiaoyu): Add dict of Parameter id to Parameter Block, for using id
        # to query Parameter Block.
        self._blocks[name] = Block("", name, module)

    def __setattr__(self, name: str, value=None):
        if isinstance(value, Module):
            self._add_block(name, value)
        elif isinstance(value, Optimizer):
            raise AttributeError(
                "'{}' object are not allowed to set Optimizer attribute named '{}', "
                "please use add_optimizer(...) instead.".format(
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
                mod_str = add_indent(mod_str, 2)
                child_lines.append(mod_str)
            lines = child_lines

        main_str = "(" + self._name + ":" + self.__class__.__name__ + ":GRAPH): ("
        if lines is not None:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


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
        if self.proto.has_train_conf():
            return True
        if self.proto.has_predict_conf():
            return False
        raise NotImplementedError

    def _train(self, mode: bool = True):
        if mode:
            self.proto.mutable_train_conf()
            self.proto.mutable_train_conf().set_loss_scale_factor(1.0)
        else:
            self.proto.mutable_predict_conf()

    def add_optimizer_config(
        self, optimizer_config: OptimizerConfig = None, var2var_op_name: Dict = None
    ):
        optimizer_config.optimizer.add_to_graph_train_config(
            self.proto.mutable_train_conf(), var2var_op_name
        )
