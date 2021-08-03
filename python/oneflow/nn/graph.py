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
from typing import Dict

import oneflow._oneflow_internal
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.graph_build_util as graph_build_util
import oneflow.framework.session_context as session_ctx
from oneflow.framework.tensor import Tensor
from oneflow.framework.function_util import FunctionConfig
from oneflow.framework.multi_client_session import MultiClientSession
from oneflow.framework.tensor_tuple_util import convert_to_tensor_tuple
from oneflow.nn.graph_block import Block, BlockType
from oneflow.nn.graph_optimizer import OptimizerConfig, VariableConfig
from oneflow.nn.module import Module
from oneflow.nn.optimizer.optimizer import Optimizer
from oneflow.nn.util import add_indent


class Graph(object):
    _child_init_cnt = dict()

    def __init__(self):
        self.config = GraphConfig()
        self._generate_name()
        self.config.proto.set_job_name(self._name)
        self._c_nn_graph = oneflow._oneflow_internal.nn.graph.CNNGraph(self._name)
        self._blocks = OrderedDict()
        self._optimizers_conf = OrderedDict()
        self._variables_conf = OrderedDict()
        self._is_compiled = False
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
        self._optimizers_conf[name] = OptimizerConfig(
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

    def _generate_optimizer_and_variable_configs(self):
        if len(self._optimizers_conf) > 0:
            self.config._train(True)
        for state_block in self._state():
            if state_block.type == BlockType.PARAMETER:
                self._variables_conf[state_block.origin] = VariableConfig(
                    state_block.name_prefix + state_block.name
                )
        for name, opt_config in self._optimizers_conf.items():
            self.config._generate_optimizer_and_variable_configs(
                opt_config, self._variables_conf
            )

    def _compile(self, *args):
        assert not self._is_compiled, (
            "nn.Graph " + self._name + " has already been compiled."
        )

        self._generate_optimizer_and_variable_configs()

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
                if state_block.type == BlockType.PARAMETER:
                    state_config = self._variables_conf[state_block.origin]
                else:
                    state_config = None
                state_block.set_lazy_origin_builder(
                    partial(
                        graph_build_util.build_graph_state,
                        op_name,
                        state_tensor,
                        state_config,
                    )
                )

            self._variables = convert_to_tensor_tuple(state_tensors)

            # Deal with module in self.build(*args)
            outputs = self.build(*lazy_args)

            # Deal with outputs
            if not (type(outputs) is tuple or type(outputs) is list):
                if outputs is None:
                    outputs = ()
                else:
                    assert type(outputs) is Tensor
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

            self._outputs = convert_to_tensor_tuple(eager_outputs)
            self._eager_outputs = eager_outputs

            # Register input/output/variable to _c_nn_graph
            self._c_nn_graph.register_input_op_names(lazy_arg_op_names)
            self._c_nn_graph.register_output_op_names(eager_output_op_names)
            self._c_nn_graph.register_variable_op_names_and_tensors(
                state_op_names, self._variables
            )

            # Save job proto for debug
            self._job_proto = c_api_util.GetCurrentJob()

        # Complie and init Runtime
        self._c_nn_graph.complie_and_init_runtime()
        self._is_compiled = True
        return eager_outputs

    def _launch(self, *args):
        # oneflow._oneflow_internal.eager.multi_client.Sync() NOTE(chengcheng): Need Sync?
        oneflow._oneflow_internal.nn.graph.RunLazyNNGraph(
            convert_to_tensor_tuple(args),
            self._outputs,
            self._variables,
            self._c_nn_graph,
        )
        return self._eager_outputs

    def __call__(self, *args):
        if not self._is_compiled:
            self._compile(*args)
        return self._launch(*args)

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

    def _generate_optimizer_and_variable_configs(
        self,
        optimizer_config: OptimizerConfig = None,
        variables_conf: OrderedDict = None,
    ):
        optimizer_config.generate_optimizer_and_variable_configs(
            self.proto.mutable_train_conf(), variables_conf
        )


from oneflow.nn.graph import Graph as Graph
from oneflow.nn.graph_block import Block, BlockConfig
from oneflow.nn.graph_optimizer import OptimizerConfig
