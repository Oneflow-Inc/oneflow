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
from oneflow.framework.distribute import get_rank
from oneflow.framework.tensor import Tensor, TensorTuple
from oneflow.framework.multi_client_session import MultiClientSession
from oneflow.framework.tensor_tuple_util import convert_to_tensor_tuple
from oneflow.nn.graph.block import Block, BlockType
from oneflow.nn.graph.config import GraphConfig
from oneflow.nn.graph.optimizer import OptDict, VariableConfig
from oneflow.amp import GradScaler
from oneflow.nn.graph.util import add_indent, sys_exc_error_msg, list_to_func_return
from oneflow.nn.module import Module
from oneflow.nn.optimizer.optimizer import Optimizer
from oneflow.nn.optimizer.lr_scheduler import LrScheduler


class Graph(object):
    _child_init_cnt = dict()

    def __init__(self):
        self._generate_name()
        self.config = GraphConfig()
        self._blocks = OrderedDict()
        self._opts = []
        self._grad_scaler = None
        self._variables_conf = OrderedDict()
        self._is_compiled = False
        self._job_proto = None
        self._args_repr = []
        self._outs_repr = []
        self._debug = False
        self._c_nn_graph = oneflow._oneflow_internal.nn.graph.CNNGraph(self._name)
        session = session_ctx.GetDefaultSession()
        assert type(session) is MultiClientSession
        session.TryInit()
        session.AddCGraph(self._c_nn_graph)

    @property
    def name(self):
        return self._name

    @property
    def training(self):
        return self.config.training

    @property
    def _config_proto(self):
        return self.config.proto

    @property
    def _optimization_conf_proto(self):
        session = session_ctx.GetDefaultSession()
        assert type(session) is MultiClientSession
        return session.resource

    @property
    def _graph_proto(self):
        return self._job_proto

    def debug(self, mode: bool = True) -> None:
        if get_rank() != 0:
            return
        else:
            print("Note that nn.Graph.debug() only print debug info on rank 0.")
        self._debug = mode
        for name, block in self._blocks.items():
            assert block.type == BlockType.MODULE
            block.debug(mode)

    def build(self, *args):
        raise NotImplementedError()

    def add_optimizer(
        self, optim: Optimizer, *, lr_sch: LrScheduler = None,
    ):
        opt_dict = dict()
        assert optim is not None, "optimizer cannot be None"
        assert isinstance(
            optim, Optimizer
        ), "optimizer must be an instance of Optimizer"
        opt_dict["optim"] = optim
        if lr_sch is not None:
            assert isinstance(lr_sch, LrScheduler)
            assert (
                lr_sch._optimizer is optim
            ), "lr_scheduler's optimizer must be the same optimizer in add_optimizer."
            opt_dict["lr_sch"] = lr_sch
        self._opts.append(opt_dict)

    def set_grad_scaler(self, grad_scaler: GradScaler = None):
        assert isinstance(grad_scaler, GradScaler)
        self._grad_scaler = grad_scaler

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

    def _generate_config_proto(self):
        self.config.proto.set_job_name(self._name)

        if self._grad_scaler is not None:
            self._grad_scaler.generate_conf_for_graph(
                self.config.proto.mutable_train_conf()
            )

        if len(self._opts) > 0:
            self.config._train(True)
        for state_block in self._state():
            if state_block.type == BlockType.PARAMETER:
                self._variables_conf[state_block.origin] = VariableConfig(
                    state_block.name_prefix + state_block.name
                )
        for opt in self._opts:
            opt_dict = OptDict(opt)
            self.config._generate_optimizer_and_variable_configs(
                opt_dict, self._variables_conf
            )

    def _compile(self, *args):
        # Build forward graph
        try:
            if self._debug:
                print(self._shallow_repr() + " start building forward graph.")
            assert not self._is_compiled, (
                "nn.Graph " + self._name + " has already been compiled."
            )

            eager_outputs = self._build_forward_graph(*args)

            if self._debug:
                print(self._shallow_repr() + " end building forward graph.")
        except:
            print(
                "[ERROR]"
                + self._shallow_repr()
                + " build forward graph got error: "
                + sys_exc_error_msg()
            )
            raise

        # Complie and init Runtime
        try:
            if self._debug:
                print(self._shallow_repr() + " start compiling and init graph runtime.")

            self._c_nn_graph.complie_and_init_runtime()

            if self._debug:
                print(self._shallow_repr() + " end compiling and init graph rumtime.")
        except:
            print(
                "[ERROR]"
                + self._shallow_repr()
                + " compiling and initialing graph runtime got error : ",
                sys_exc_error_msg(),
            )
            raise

        self._is_compiled = True
        return eager_outputs

    def _build_forward_graph(self, *args):
        session = session_ctx.GetDefaultSession()
        assert type(session) is MultiClientSession
        self._generate_config_proto()
        with graph_build_util.graph_build_context(self.config.proto, session):
            # Deal with inputs
            arg_op_names, lazy_args, self._args_repr = self._build_io(
                "input", graph_build_util.build_graph_input_arg, *args
            )

            # Deal with parameter and buffer
            state_op_names, self._states_tensor_tuple = self._build_states()

            # Deal with module in self.build(*args)
            outputs = self.build(*lazy_args)

            # Deal with outputs
            if not (type(outputs) is tuple or type(outputs) is list):
                if outputs is None:
                    outputs = ()
                else:
                    outputs = (outputs,)
            output_op_names, self._eager_outputs, self._outs_repr = self._build_io(
                "output", graph_build_util.build_graph_output, *outputs
            )
            self._outputs_tensor_tuple = convert_to_tensor_tuple(
                self._flatten_io("output", *self._eager_outputs)
            )
            self._eager_outputs = list_to_func_return(self._eager_outputs)

            # Register input/output/variable to _c_nn_graph
            self._c_nn_graph.register_input_op_names(arg_op_names)
            self._c_nn_graph.register_output_op_names(output_op_names)
            self._c_nn_graph.register_variable_op_names_and_tensors(
                state_op_names, self._states_tensor_tuple
            )

            # Save job proto for debug
            self._job_proto = c_api_util.GetCurrentJob()

        return self._eager_outputs

    def _run(self, *args):
        try:
            flattened_eager_args = self._flatten_io("input", *args)
            # oneflow._oneflow_internal.eager.multi_client.Sync() NOTE(chengcheng): Need Sync?
            oneflow._oneflow_internal.nn.graph.RunLazyNNGraph(
                convert_to_tensor_tuple(flattened_eager_args),
                self._outputs_tensor_tuple,
                self._states_tensor_tuple,
                self._c_nn_graph,
            )
        except:
            print(
                "[ERROR]"
                + self._shallow_repr()
                + " run got error : "
                + sys_exc_error_msg()
            )
            raise
        return self._eager_outputs

    def __call__(self, *args):
        if not self._is_compiled:
            self._compile(*args)

        return self._run(*args)

    def _build_io(self, io_type, build_func, *args):
        assert io_type in ("input", "output")
        io_type_upper = io_type.upper()
        build_args = []
        op_names = []
        args_repr = []

        def build_tensor_or_none(tensor, name, repr_str):
            assert tensor is None or (isinstance(tensor, Tensor))
            if isinstance(tensor, Tensor):
                build_arg = build_func(name, tensor)
                op_names.append(name)
            else:
                build_arg = None

            args_repr.append(repr_str)
            if self._debug:
                print(repr_str)
            return build_arg

        for idx, arg in enumerate(args):
            if isinstance(arg, Tensor) or arg is None:
                if arg is None:
                    name, repr_str = self._io_item_check_and_gen(
                        arg, None, io_type, idx
                    )
                else:
                    name, repr_str = self._io_item_check_and_gen(
                        arg, Tensor, io_type, idx
                    )
                build_args.append(build_tensor_or_none(arg, name, repr_str))
            elif isinstance(arg, (TensorTuple, list)):
                if isinstance(arg, TensorTuple):
                    seq_args = TensorTuple()
                else:
                    seq_args = list()
                for i in range(len(arg)):
                    name, repr_str = self._io_item_check_and_gen(
                        arg[i], Tensor, io_type, idx, i
                    )
                    seq_args.append(build_tensor_or_none(arg[i], name, repr_str))
                build_args.append(seq_args)
            else:
                self._io_item_check_and_gen(arg, Tensor, io_type, idx)

        return op_names, build_args, args_repr

    def _flatten_io(self, io_type, *args):
        assert isinstance(args, tuple)
        flattened_args = []
        for idx, arg in enumerate(args):
            if isinstance(arg, Tensor):
                flattened_args.append(arg)
            elif isinstance(arg, (TensorTuple, list)):
                for i in range(len(arg)):
                    self._io_item_check(arg[i], Tensor, io_type, idx, i)
                    flattened_args.append(arg[i])
            else:
                self._io_item_check(arg, None, io_type, idx)
        return flattened_args

    def _io_item_check(self, item, expect_type, io_type, idx, second_idx=None):
        if expect_type is None and item is None:
            return
        elif expect_type is not None and isinstance(item, expect_type):
            return
        else:
            assert io_type in ("input", "output")
            name = (
                "_"
                + self.name
                + "-"
                + io_type
                + "_"
                + str(idx)
                + ("" if second_idx is None else "_" + str(second_idx))
            )
            repr_str = (
                "[ERROR](" + io_type.upper() + ":" + name + ":" + str(type(item)) + ")"
            )
            print(repr_str)
            raise NotImplementedError(
                "nn.Graph.build()'s input/output only support types: Tensor/TensorTuple/list(Tensor)/None."
            )

    def _io_item_check_and_gen(self, item, expect_type, io_type, idx, second_idx=None):
        assert io_type in ("input", "output")
        name = (
            "_"
            + self.name
            + "-"
            + io_type
            + "_"
            + str(idx)
            + ("" if second_idx is None else "_" + str(second_idx))
        )
        if expect_type is None and item is None:
            repr_str = (
                "[WARNING]("
                + io_type.upper()
                + ":"
                + name
                + ":"
                + str(type(item))
                + ")"
            )
            return name, repr_str
        elif expect_type is not None and isinstance(item, expect_type):
            if isinstance(item, Tensor):
                repr_str = (
                    "(" + io_type.upper() + ":" + name + ":" + item._meta_repr() + ")"
                )
            else:
                repr_str = (
                    "[WARNING]("
                    + io_type.upper()
                    + ":"
                    + name
                    + ":"
                    + str(type(item))
                    + ")"
                )
            return name, repr_str
        else:
            repr_str = (
                "[ERROR](" + io_type.upper() + ":" + name + ":" + str(type(item)) + ")"
            )
            print(repr_str)
            raise NotImplementedError(
                "nn.Graph.build()'s input/output only support types: Tensor/TensorTuple/list(Tensor)/None."
            )

    def _build_states(self):
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
        state_tensor_tuple = convert_to_tensor_tuple(state_tensors)
        return state_op_names, state_tensor_tuple

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
        child_lines = []
        if len(self._args_repr) > 0:
            for in_str in self._args_repr:
                input_str = add_indent(in_str, 2)
                child_lines.append(input_str)

        if len(self._blocks) > 0:
            for n, m in self._blocks.items():
                mod_str = repr(m)
                mod_str = add_indent(mod_str, 2)
                child_lines.append(mod_str)

        if len(self._outs_repr) > 0:
            for out_str in self._outs_repr:
                output_str = add_indent(out_str, 2)
                child_lines.append(output_str)

        main_str = self._shallow_repr() + ": ("
        if len(child_lines) > 0:
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str

    def _shallow_repr(self):
        shallow_repr = "(GRAPH:" + self._name + ":" + self.__class__.__name__ + ")"
        return shallow_repr
