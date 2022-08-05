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
import logging
import os
import time
import inspect
from collections import OrderedDict
from functools import partial
from typing import Dict, Optional, Union, List, Callable
import weakref
from google.protobuf import text_format

import oneflow
import oneflow._oneflow_internal
import oneflow.core.job.job_pb2 as job_pb
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.graph_build_util as graph_build_util
import oneflow.framework.session_context as session_ctx
from oneflow.amp import GradScaler, StaticGradScaler
from oneflow.env import get_rank
from oneflow.framework.multi_client_session import MultiClientSession
from oneflow.framework.tensor import Tensor, TensorTuple
from oneflow.framework.tensor_tuple_util import convert_to_tensor_tuple
from oneflow.nn.graph.block import Block, BlockType, get_block_cls
from oneflow.nn.graph.graph_config import GraphConfig
from oneflow.nn.graph.optimizer import OptDict, VariableConfig
from oneflow.nn.graph.util import (
    add_indent,
    ArgsTree,
    operators_repr,
    seq_to_func_return,
    sys_exc_error_msg,
)
from oneflow.nn.module import Module
from oneflow.nn.optimizer.lr_scheduler import LRScheduler
from oneflow.nn.optimizer.optimizer import Optimizer


class Graph(object):
    r"""Base class for training or evaluating a neural network in static graph mode.

    To use static graph mode for model training or evaluation in OneFlow, you should:

    1. Define your customized graph as a subclass of ``nn.Graph``.
    2. Add ``super().__init__()`` in your subclass's ``__init__()``.
    3. Add modules to your graph as regular attributes.
    4. Define computation logical in ``build()`` method.
    5. Instantiate your graph then call it.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> class LinearGraph(flow.nn.Graph):
        ...    def __init__(self):
        ...        super().__init__()
        ...        # Add a module to the graph.
        ...        self.linear = flow.nn.Linear(3, 8, False)
        ...    def build(self, x):
        ...        # Use the module to build the computation logic of the graph.
        ...        return self.linear(x)

        # Instantiate the graph
        >>> linear_graph = LinearGraph()
        >>> x = flow.randn(4, 3)

        # First call on graph will run graph's build() method to
        # trace a computatioin graph. Then the computation graph will be
        # optimized and executed for the first time.
        >>> linear_graph(x).shape
        oneflow.Size([4, 8])

        # Later call on graph will execute the computation graph directly.
        >>> linear_graph(x).shape
        oneflow.Size([4, 8])

    Note:
        nn.Graph cannot be nested at the moment.
    """
    _child_init_cnt = dict()

    def __init__(self):
        """
        Initializes internal Graph states. It MUST be called in ``__init__`` method of subclass.

        For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> class SubclassGraph(flow.nn.Graph):
            ...     def __init__(self):
            ...         super().__init__() # MUST be called
            ...         # Then define the graph attributes
            ...     def build(self):
            ...         pass

        """
        self._generate_name()
        self.config = GraphConfig()
        self._blocks = OrderedDict()
        self._opts = []
        self._verbose = False
        self._grad_scaler = None
        self._variables_conf = OrderedDict()
        self._additional_variable_tobe_loaded = OrderedDict()
        self._is_compiled = False
        # Default is local view
        self._is_global_view = False
        # forward graph job proto
        self._forward_job_proto = None
        # forward, backward and optimized graph job proto
        self._full_job_proto = None
        # completed graph job proto
        self._compiled_job_proto = None
        self._job_id = None
        self._args_repr = []
        self._outs_repr = []
        self._debug = False
        self._debug_min_s_level = 2
        self._debug_max_v_level = 0
        self._debug_max_py_stack_depth = 2
        self._debug_op_repr_with_py_stack = False
        self._debug_only_user_py_stack = True
        self._outputs_buffer_size = 2
        self._cur_index_of_ouputs_buffer = 0

        # For graph level op rewrite
        self._unique_global_op_dict = dict()
        self._unique_identity_op_dict = dict()

        self._session = session_ctx.GetDefaultSession()
        assert type(self._session) is MultiClientSession
        self._session.TryInit()
        self._c_nn_graph = None
        self.env_enable_mlir_inference_opt = None

    def build(self, *args, **kwargs):
        r"""The ``build()`` method must be overridden to define neural network
        computaion logic.

        The ``build()`` method of nn.Graph is very similar to the ``forward()``
        method of nn.Module. It is used to describe the computatioin logical of
        a neural network.

        When a graph object being called for the first time, the ``build()``
        method will be called implicitly to build the computatioin graph.

        Make sure to call modules's ``train()`` or ``eval()`` method before the
        first call of your graph to make the module executing the right
        training or evaluation logic if needed.

        For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> linear = flow.nn.Linear(3, 8, False)
            >>> class MyGraph(flow.nn.Graph):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.model = linear
            ...     def build(self, x):
            ...         return self.model(x)

            >>> linear_graph = MyGraph()
            >>> x = flow.randn(4, 3)
            >>> linear.eval() # make linear module executing in evaluation mode
            Linear(in_features=3, out_features=8, bias=False)
            >>> y = linear_graph(x) # The build() method is called implicitly

        Note:
            ``build()`` method's inputs and outputs support list/tuple/dict,
            but the item in them must be one of these types:

            * ``Tensor``
            * ``None``

        """
        raise NotImplementedError(
            "nn.Graph.build() method must be overridden when subclassing the nn.Graph."
        )

    def __call__(self, *args, **kwargs):
        r"""Call nn.Graph subclass instance to run your customized graph.

        Call your customized graph after the instantiation:

        For example:

        .. code-block:: python

            g = CustomGraph()
            out_tensors = g(input_tensors)

        The inputs of ``__call__`` method must match the inputs of ``build()``
        method. And the ``__call__`` method will return outputs matching the
        outputs of ``build()`` method.

        Note:
            The first call takes longer than later calls, because nn.Graph
            will do the computaion graph generation and optimization at the first call.

            Donot override this function.
        """

        if not self._is_compiled:
            self._compile(*args, **kwargs)
            self.__print(
                0, 2, lambda: f"{self.name} with operators:\n" + self.__repr__()
            )

        return self.__run(*args, **kwargs)

    def add_optimizer(
        self, optim: Optimizer, *, lr_sch: LRScheduler = None, is_sparse: bool = False,
    ):
        r"""Add an optimizer, an learning rate scheduler to the graph.

        To do training with nn.Graph, you should do 2 more things:

        1. Add at least one optimizer(learning rate schedulers are optional) with ``add_optimizer()`` method.
        2. Call loss tensor's ``backward()`` method in ``build()`` method.

        Note that the computaion graph will automatically execute these methods:

        * optimizer's ``clip_grad()`` if a optimizer is set to do grad cliping.
        * optimizer's ``step()``.
        * optimizer's ``zero_grad()``.
        * learn rate scheduler's ``step()``.

        Also note that only scalar tensor are allowed to call ``backward()``
        in ``nn.Graph.build()`` for the moment. So you may call methods such as ``Tensor.mean()``
        to make the loss tensor a scalar tensor.

        Note:
            If you want to output the learning rate information for each step,
            set the ``verbose`` parameter of the ``lr_scheduler`` to ``True``, and you will see the result at rank 0.

            This feature is the same as eager mode.

        For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> loss_fn = flow.nn.MSELoss(reduction="sum")
            >>> model = flow.nn.Sequential(flow.nn.Linear(3, 1), flow.nn.Flatten(0, 1))
            >>> optimizer = flow.optim.SGD(model.parameters(), lr=1e-6)
            >>> class LinearTrainGraph(flow.nn.Graph):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.model = model
            ...         self.loss_fn = loss_fn
            ...         # Add an optimizer
            ...         self.add_optimizer(optimizer)
            ...     def build(self, x, y):
            ...         y_pred = self.model(x)
            ...         loss = self.loss_fn(y_pred, y)
            ...         # Call loss tensor's backward(), loss tensor must be a scalar tensor
            ...         loss.backward()
            ...         return loss

            >>> linear_graph = LinearTrainGraph()
            >>> x = flow.randn(10, 3)
            >>> y = flow.randn(10)
            >>> model.train() # make model executing in training mode
            Sequential(
              (0): Linear(in_features=3, out_features=1, bias=True)
              (1): Flatten(start_dim=0, end_dim=1)
            )
            >>> for t in range(3):
            ...     loss = linear_graph(x, y)

        Args:
            optim (oneflow.optim.Optimizer): The optimizer.
            lr_sch : The learning rate scheduler, see oneflow.optim.lr_scheduler.
            is_sparse: When set to be True, treat optim as a sparse optimizer. Default is False.
        """
        opt_dict = dict()
        assert optim is not None, "optimizer cannot be None"
        assert isinstance(
            optim, Optimizer
        ), "optimizer must be an instance of Optimizer"

        opt_dict["optim"] = optim
        opt_dict["is_sparse"] = bool(is_sparse)
        if lr_sch is not None:
            assert isinstance(lr_sch, LRScheduler)
            assert (
                lr_sch.optimizer is optim
            ), "lr_scheduler's optimizer must be the same optimizer in add_optimizer."
            opt_dict["lr_sch"] = lr_sch
            self._verbose = opt_dict["lr_sch"].verbose
            rank = get_rank()
            if rank != 0:
                self._verbose = False
        oneflow._oneflow_internal.SetGraphLRVerbose(self._verbose)
        self._opts.append(opt_dict)
        # Set the training config if there is an optimizer add in graph.
        if len(self._opts) == 1:
            self.config._train(True)

    def set_grad_scaler(self, grad_scaler: GradScaler = None):
        r"""Set the GradScaler for gradient and loss scaling."""
        assert isinstance(grad_scaler, (GradScaler, StaticGradScaler))
        self._grad_scaler = grad_scaler

    def state_dict(
        self, destination=None
    ) -> Dict[str, Union[Dict[str, Tensor], Tensor]]:
        r"""Returns a dictionary containing a whole state of the graph.

        States of modules/optimizers/lr schedulers in a graph are included.

        Keys of modules' state dict are corresponding to their name in the graph.
        Values of modules' state dict are corresponding to their nn.Module's
        state dict.

        Other keys and tensors are states of optimizers/lr schedulers/etc.

        Returns:
            dict: a dictionary containing the whole state of the graph.

        """
        # Sync to make sure states has been updated.
        oneflow._oneflow_internal.eager.Sync()
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        # Get states from sub module block
        for name, block in self._blocks.items():
            assert block.type == BlockType.MODULE
            sub_destination = OrderedDict()
            sub_destination._metadata = OrderedDict()
            module = block.origin
            if module is not None:
                module.state_dict(
                    sub_destination, "", keep_vars=False,
                )
            destination[name] = sub_destination
        # Get additional states.
        # Additional variables are states in Optimizer/LRScheduler and free eager tensors of nn.Graph.
        if self._is_compiled:
            # Get from _c_nn_graph.
            additional_var_names = self._c_nn_graph.additional_var_names
            additional_var_tensors = self._c_nn_graph.additional_var_tensors
            assert len(additional_var_names) == len(additional_var_tensors)
            for i in range(len(additional_var_names)):
                additional_tensor = additional_var_tensors[i]
                if not self._is_global_view:
                    additional_tensor = additional_tensor.to_local()
                destination[additional_var_names[i]] = additional_tensor
        else:
            # Get from loaded dict.
            for name, item in self._additional_variable_tobe_loaded.items():
                destination[name] = item
        return destination

    def load_state_dict(
        self,
        state_dict: Dict[str, Union[Dict[str, Tensor], Tensor]],
        strict: bool = True,
    ):
        r"""Copies module's states and other graph states from :attr:`state_dict`
        into this graph. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`nn.Graph.state_dict` function.

        Args:
            state_dict (dict): a dict containing module's states and other graph states.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this graph's
                :meth:`nn.Graph.state_dict` function. Default: ``True``.

        Note:
            nn.Graph's state dict can only be loaded before the first call of a graph.
        """
        assert (
            not self._is_compiled
        ), "nn.Graph's state dict can only be loaded before the first call of a graph."
        # Additional variables are states in Optimizer or LRScheduler of nn.Graph.
        for name, item in state_dict.items():
            if name in self._blocks:
                # 1 load parameter/buffer to Modules
                self._blocks[name].origin.load_state_dict(item, strict)
            else:
                # 2 store other state to CNNGraph, CNNGraph load them after job pass
                assert isinstance(item, Tensor)
                self._additional_variable_tobe_loaded[name] = item

    @property
    def name(self):
        r"""Name auto-generated for this graph."""
        return self._name

    @property
    def is_compiled(self):
        r"""Whether this graph is compiled or not
        """
        return self._is_compiled

    @property
    def training(self):
        r"""In traninig mode if the graph has an optimizer."""
        return self.config.training

    def debug(
        self,
        v_level: int = -1,
        *,
        ranks: Optional[Union[int, List[int]]] = None,
        max_py_stack_depth: int = 2,
        only_user_py_stack=True,
        op_repr_with_py_stack=False,
    ) -> None:
        r"""Open or close debug mode of the graph.

        If in debug mode, logs of computation graph building infos or warnings will be
        printed. Otherwise, only errors will be printed.

        Each nn.Module inside a nn.Graph also has a debug() method to enable debug mode.

        Use ``v_level`` to choose verbose debug info level, default level is 0, max level is 3.
        ``v_level`` -1 will disable the debug mode of the graph (i.e. no info will be printed).
        ``v_level`` 0 will print warning and graph building stages. ``v_level`` 1 will additionally
        print graph build info of each nn.Module. ``v_level`` 2 will additionally print graph build
        info of each operation. ``v_level`` 3 will additionally print more detailed info of each
        operation.

        Use ``ranks`` to choose which rank to print the debug information.

        Use ``max_py_stack_depth`` to specify the max Python stack depth for the debug information.
        
        Use ``only_user_py_stack`` to only print the operators' locations which are from users' code or models.

        Use ``op_repr_with_py_stack`` to print operators' locations when printing nn.Graph's repr.

        For example:

        .. code-block:: python

            g = CustomGraph()
            g.debug()  # Open debug mode
            out_tensors = g(input_tensors)  # Will print log for debug at the first call

        Args:
            v_level (int): choose verbose debug info level, default v_level is 0, max v_level is 3. v_level can be set to -1 to close the debug mode.
            ranks (int or list(int)): choose ranks to print the debug information. Default rank ``0``.
                You can choose any valid rank. Ranks equals ``-1`` means debug on all ranks.
            max_py_stack_depth(int): the maximum depth for the Python stack debug information. Default: ``2``.
            only_user_py_stack(bool): only to print the operators' locations from users' code. Default: ``True``.
            op_repr_with_py_stack(bool):  print operators' locations when printing nn.Graph's repr. Default: ``False``. 
        """
        assert isinstance(v_level, int)
        assert v_level >= -1, "The min verbose debug info level is -1."
        assert v_level <= 3, "The max verbose debug info level is 3."
        assert max_py_stack_depth >= 0, "The min max stack depth is 0."
        assert isinstance(max_py_stack_depth, int)
        assert isinstance(only_user_py_stack, bool)
        assert isinstance(op_repr_with_py_stack, bool)

        if ranks is None:
            rank_list = [0]
        elif isinstance(ranks, int):
            rank_list = [ranks]
        elif isinstance(ranks, list):
            rank_list = ranks
        else:
            raise ValueError("ranks must be int or List[int].")

        my_rank = get_rank()
        if -1 in rank_list or my_rank in rank_list:
            self._debug = v_level >= 0
            if self._debug:
                self._debug_min_s_level = 0
                self._debug_max_v_level = max(0, v_level)
            for name, block in self._blocks.items():
                assert block.type == BlockType.MODULE
                block.debug(
                    v_level,
                    ranks=ranks,
                    max_py_stack_depth=max_py_stack_depth,
                    only_user_py_stack=only_user_py_stack,
                    op_repr_with_py_stack=op_repr_with_py_stack,
                )

        self._debug_max_py_stack_depth = max_py_stack_depth
        self._debug_op_repr_with_py_stack = op_repr_with_py_stack
        self._debug_only_user_py_stack = only_user_py_stack

    def __repr__(self):
        r"""For printing the graph structure.

        The graph structure can be printed after graph instantiation.

        After the first call of graph, inputs and outputs will be added to
        the graph structure.

        For example:

        .. code-block:: python

            g = CustomGraph()
            print(g)

            out_tensors = g(input_tensors)
            print(g) # Inputs and Outputs infos are added

        """
        child_lines = []
        child_lines.append(add_indent(repr(self.config), 2))
        if len(self._args_repr) > 0:
            for in_str in self._args_repr:
                input_str = add_indent(in_str, 2)
                child_lines.append(input_str)

        if len(self._blocks) > 0:
            for n, m in self._blocks.items():
                mod_str = repr(m)
                mod_str = add_indent(mod_str, 2)
                child_lines.append(mod_str)

        for op_str in self._ops_repr():
            child_lines.append(add_indent(op_str, 2))

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

    def _ops_repr(self):
        r"""Generate operators' string representation of this graph
        """
        if self._is_compiled and self._compiled_graph_proto is not None:
            module_conf = self._compiled_graph_proto.module_name2module_conf[self.name]
            return operators_repr(
                module_conf.ops,
                self._compiled_graph_proto,
                self._debug_op_repr_with_py_stack,
            )

        return []

    def __print(self, s_level=2, v_level=0, msg=None):
        r"""Do print according to info level."""
        assert isinstance(s_level, int)
        assert isinstance(v_level, int)
        assert isinstance(msg, str) or isinstance(msg, Callable)
        if s_level >= self._debug_min_s_level:
            if (s_level > 0) or (s_level == 0 and v_level <= self._debug_max_v_level):
                if isinstance(msg, str):
                    print(msg, flush=True)
                elif isinstance(msg, Callable):
                    print(msg(), flush=True)

    @property
    def _config_proto(self):
        return self.config.proto

    @property
    def _optimization_conf_proto(self):
        return self._session.resource

    @property
    def _graph_proto(self):
        if not self._is_compiled:
            self.__print(
                2,
                0,
                f"[ERROR]{self._shallow_repr()} has not been compiled, so it's graph proto is None."
                " You can call the graph to trigger it's compilation.",
            )
        return self._forward_job_proto

    @property
    def _full_graph_proto(self):
        if self._full_job_proto is None:
            self.__print(
                2,
                0,
                f"[ERROR]{self._shallow_repr()} has not been compiled, so it's full graph proto is None."
                " You can call the graph to trigger it's compilation.",
            )
        return self._full_job_proto

    @_full_graph_proto.setter
    def _full_graph_proto(self, full_job_proto):
        assert (
            not self._is_compiled
        ), "nn.Graph's full graph proto can only be set before the first compilation."
        self._full_job_proto = full_job_proto
        self._c_nn_graph.job = full_job_proto.SerializeToString()

    @property
    def _compiled_graph_proto(self):
        if not self._is_compiled:
            self.__print(
                2,
                0,
                f"[ERROR]{self._shallow_repr()} has not been compiled, so it's compiled graph proto is None."
                " You can call the graph to trigger it's compilation.",
            )
        return self._compiled_job_proto

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

    def __ensure_state_tensors_contiguous(self):
        for state_block in self._state():
            state_tensor = state_block.origin
            if not state_tensor.is_contiguous():
                state_tensor.contiguous_()

    def _filter_states(self):
        state_tensor_set = set()
        state_tensors = []
        state_op_names = []

        for state_block in self._state():
            state_tensor = state_block.origin
            # If any state tensor is global tensor, graph is in global view.
            if state_tensor.is_global:
                self._is_global_view = True
            if state_tensor in state_tensor_set:
                continue
            op_name = state_block.name_prefix + state_block.name
            state_tensor_set.add(state_tensor)
            state_tensors.append(state_tensor)
            state_op_names.append(op_name)

            if state_block.type == BlockType.PARAMETER:
                self._variables_conf[state_tensor] = VariableConfig(op_name)

        self._state_tensor_tuple = convert_to_tensor_tuple(state_tensors)
        return state_op_names

    def _generate_config_proto(self):
        self.config.proto.job_name = self._name
        self._outputs_buffer_size = self.config._outputs_buffer_size

        if self._grad_scaler is not None:
            self._grad_scaler._generate_conf_for_graph(self.config.proto.train_conf)

        for opt in self._opts:
            opt_dict = OptDict(opt)
            self.config._generate_optimizer_and_variable_configs(
                opt_dict, self._variables_conf
            )

    def _create_states_builder(self):
        state2lazy_builder = dict()
        for state_block in self._state():
            state_tensor = state_block.origin
            op_name = state_block.name_prefix + state_block.name
            if state_tensor in state2lazy_builder:
                # Differe tensor block shares the same tensor, so they need to share the same
                # builder.
                state_block.set_lazy_origin_builder(state2lazy_builder[state_tensor])
            else:
                if state_block.type == BlockType.PARAMETER:
                    assert state_tensor in self._variables_conf
                    state_config = self._variables_conf[state_tensor]
                    op_name = state_config.name
                else:
                    state_config = None
                # Init a new lazy tensor builder
                state_block.lazy_origin_builder().name = op_name
                state_block.lazy_origin_builder().method = partial(
                    graph_build_util.build_graph_state,
                    op_name,
                    state_tensor,
                    state_config,
                )
                state2lazy_builder[state_tensor] = state_block.lazy_origin_builder()

    @staticmethod
    def to_graph(func):
        """Make a function to do static graph run with nn.Graph.

        After decorating a function with ``to_graph``, the function is turned into a naive `nn.Graph`.

        Note:
            This is just a quick way to run a simple function with nn.Graph.
            If you want to do training or model save/load, customize a nn.Graph class instead, donot use ``to_graph``.

        For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> @flow.nn.Graph.to_graph
            ... def test_func(x):
            ...     return x * 2
            >>> input = flow.tensor((1, 2), dtype=flow.float32)
            >>> out = test_func(input)
            >>> out
            tensor([2., 4.], dtype=oneflow.float32)

        ..
            Feature Stage of Feature [to_graph].
            - Maintainer List [@strint]
            - Current Stage [Pre-alpha, note that this is an experimental feature and maybe removed without notice.]

        """
        assert inspect.isfunction(
            func
        ), f"nn.Graph.to_graph only support function currently, so {func} must be a function."
        graph_cls_name = func.__name__ + "_graph"

        def init(self):
            super(graph_cls_name, self).__init__()

        def build(self, *args, **kwargs):
            return func(*args, **kwargs)

        graph_cls_name = type(
            graph_cls_name, (Graph,), {"__init__": init, "build": build,},
        )

        a_graph = graph_cls_name()

        return a_graph

    def _compile(self, *args, **kwargs):
        self.__ensure_input_tensors_contiguous(*args, **kwargs)
        _, eager_outputs = self.build_graph(*args, **kwargs)
        self.finish_complie_and_init_runtime()
        return eager_outputs

    def build_graph(self, *args, **kwargs):
        # Build graph
        try:
            self.__print(0, 0, self._shallow_repr() + " start building graph.")
            assert not self._is_compiled, (
                "nn.Graph " + self._name + " has already been compiled."
            )
            build_graph_start = time.perf_counter()
            with graph_build_util.DebugScopeContext(
                self._debug_min_s_level,
                self._debug_max_v_level,
                self._debug,
                self._debug_max_py_stack_depth,
                self._debug_only_user_py_stack,
            ):
                outputs = self.__build_graph(*args, **kwargs)
            build_graph_end = time.perf_counter()
            self.__print(
                0,
                0,
                self._shallow_repr()
                + " building graph Done! Cost time: "
                + str(round(build_graph_end - build_graph_start, 2))
                + "s."
                + "\n",
            )
            return outputs
        except:
            self.__print(
                2, 0, "[ERROR]" + self._shallow_repr() + " building graph got error."
            )
            raise

    def finish_complie_and_init_runtime(self):
        additional_var_names = list()
        additional_var_tensors = list()
        for name, tensor in self._additional_variable_tobe_loaded.items():
            additional_var_names.append(name)
            additional_var_tensors.append(tensor)
        if len(additional_var_names) > 0:
            self._c_nn_graph.register_additional_variable_names_and_tensors(
                additional_var_names, convert_to_tensor_tuple(additional_var_tensors)
            )
        # Sync to make sure states has been loaded.
        oneflow._oneflow_internal.eager.Sync()

        # Complie graph to execution plan and init Runtime
        try:
            self.__print(
                0, 0, self._shallow_repr() + " start building plan.",
            )
            compile_and_init_start = time.perf_counter()
            with graph_build_util.DebugScopeContext(
                self._debug_min_s_level,
                self._debug_max_v_level,
                self._debug,
                self._debug_max_py_stack_depth,
                self._debug_only_user_py_stack,
            ):
                self._c_nn_graph.complie_and_init_runtime()
            # Get compiled job
            compiled_job_str = self._c_nn_graph.get_current_job_str()
            self._compiled_job_proto = job_pb.Job()
            self._compiled_job_proto.ParseFromString(compiled_job_str)

            compile_and_init_end = time.perf_counter()
            self.__print(
                0,
                0,
                self._shallow_repr()
                + " building plan Done! Cost time: "
                + str(round(compile_and_init_end - compile_and_init_start, 2))
                + "s."
                + "\n",
            )
        except:
            self.__print(
                2, 0, "[ERROR]" + self._shallow_repr() + " building plan got error."
            )
            raise

        self._is_compiled = True
        # After compile, _additional_variable_tobe_loaded is useless.
        self._additional_variable_tobe_loaded.clear()

    def __build_graph(self, *args, **kwargs):
        self.__ensure_state_tensors_contiguous()

        # Filter to get unique states in graph
        state_op_names = self._filter_states()

        self._generate_config_proto()

        # Deal with parameter and buffer
        self.__print(
            0,
            1,
            self._shallow_repr()
            + " start building graph builders of parameters and buffers.",
        )
        self._create_states_builder()
        self.__print(
            0,
            1,
            self._shallow_repr()
            + " end building graph builders of parameters and buffers.",
        )

        with graph_build_util.graph_build_context(self.config.proto, self._session):
            # Deal with inputs
            self.__print(0, 1, self._shallow_repr() + " start building graph inputs.")
            arg_op_names, lazy_args, lazy_kwargs, self._args_repr, _ = self.__build_io(
                "input", graph_build_util.build_graph_input_arg, *args, **kwargs
            )
            self.__print(0, 1, self._shallow_repr() + " end building graph inputs.")

            # Deal with module in self.build(*args)
            self.__print(0, 1, self._shallow_repr() + " start building graph modules.")
            outputs = self.build(*lazy_args, **lazy_kwargs)
            self.__print(0, 1, self._shallow_repr() + " end building graph modules.")

            # Deal with outputs
            self.__print(0, 1, self._shallow_repr() + " start building graph outputs.")
            # Always pack output to remain type of outputs
            outputs = (outputs,)

            (
                output_op_names,
                self._eager_outputs,
                _,  # empty kwargs return
                self._outs_repr,
                out2name,
            ) = self.__build_io("output", graph_build_util.build_graph_output, *outputs)

            self.__print(0, 1, self._shallow_repr() + " end building graph outputs.")

            # Save forward graph job proto
            self._forward_job_proto = c_api_util.GetCurrentJob()

            self.__print(
                0,
                1,
                self._shallow_repr() + " start building graph with compile passes.",
            )
            self.env_enable_mlir_inference_opt = os.getenv(
                "ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"
            )
            enable_mlir_inference_opt = (
                False
                if self.env_enable_mlir_inference_opt is None
                else bool(self.env_enable_mlir_inference_opt)
            )
            modules_has_training = False
            for item in self._blocks.values():
                if item._origin.training:
                    modules_has_training = True
                    break
            if (
                modules_has_training or self.training or self._is_global_view
            ) and enable_mlir_inference_opt:
                if self.training:
                    logging.warning(
                        "environment variable ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION will be ignored in training mode."
                    )

                if modules_has_training and not self.training:
                    logging.warning(
                        "environment variable ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION will be ignored when not all modules in graph are in eval mode. "
                    )

                if self._is_global_view:
                    logging.warning(
                        "environment variable ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION will be ignored in global mode. "
                    )
                enable_mlir_inference_opt = False
                del os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"]
            oneflow._oneflow_internal.FillVariableTensorMgr(
                state_op_names, self._state_tensor_tuple
            )
            # Complete the graph job proto
            oneflow._oneflow_internal.CurJobBuildAndInferCtx_Complete()
            # Save full graph job proto after job Complete for find real output blob shape and build it.
            self._full_job_proto = c_api_util.GetCurrentJob()
            self._job_id = (
                oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobId()
            )
            self.__print(
                0, 1, self._shallow_repr() + " end building graph with compile passes."
            )

            # Re-build outputs accoring to full graph and outputs buffer config.
            self.__print(
                0,
                1,
                self._shallow_repr()
                + " start re-building graph outputs for optimizatioin.",
            )
            self.__rebuild_outputs(out2name)
            self.__print(
                0,
                1,
                self._shallow_repr()
                + " end re-building graph outputs for optimizatioin.",
            )
            self._c_nn_graph = oneflow._oneflow_internal.nn.graph.CNNGraph(
                self._name,
                self._full_job_proto.SerializeToString(),
                self._job_id,
                self._session._session_ctx,
            )
            # Register input/output/variable/buffer to _c_nn_graph
            self._c_nn_graph.register_input_op_names_and_tensors(
                arg_op_names,
                convert_to_tensor_tuple(self.__flatten_io("input", *args, **kwargs)),
            )
            self._c_nn_graph.register_output_op_names_and_tensors(
                output_op_names, self._outputs_tensor_tuple
            )
            (
                state_op_names,
                state_tensors,
            ) = oneflow._oneflow_internal.DumpVariableTensorMgr()
            self._state_tensor_tuple = convert_to_tensor_tuple(state_tensors)

            self._c_nn_graph.register_variable_op_names_and_tensors(
                state_op_names, self._state_tensor_tuple
            )

        # Clear useless dict used in graph build.
        self._unique_global_op_dict.clear()
        self._unique_identity_op_dict.clear()

        # Always pack outputs to remain type of outputs
        return (
            self._full_job_proto,
            seq_to_func_return(self._eager_outputs_buffer[0], True),
        )

    def __rebuild_outputs(self, out2name=None):
        # NOTE(chengcheng):
        #   Lazy build output eager tensors.
        #
        #   After JobBuildAndInferCtxt.Complete, the output tensor shape
        #   could be changed by JobPass, such as GradientAccumulationRewritePass.
        def build_real_output(fake_eager_out):
            lbn = out2name[fake_eager_out] + "/out"
            assert lbn in self._full_job_proto.helper.lbn2logical_blob_desc
            blob_conf = self._full_job_proto.helper.lbn2logical_blob_desc[lbn]

            shape = tuple(blob_conf.shape.dim)
            dtype = fake_eager_out.dtype

            with oneflow._oneflow_internal.lazy_mode.guard(False):
                if fake_eager_out.is_global:
                    eager_out = oneflow.empty(
                        shape,
                        dtype=dtype,
                        placement=fake_eager_out.placement,
                        sbp=fake_eager_out.sbp,
                    )
                else:
                    eager_out = oneflow.empty(
                        shape, dtype=dtype, device=fake_eager_out.device
                    )

            return eager_out

        def convert_to_synced_tensor_tuple(*args):
            tensor_tuple = convert_to_tensor_tuple(*args)
            # tensors acting as buffer should be synced once upon created.
            oneflow._oneflow_internal.nn.graph.SoftSyncNNGraphBuffers(
                tensor_tuple, self._c_nn_graph
            )
            return tensor_tuple

        self._eager_outputs, _ = self.__map_io(
            "output", build_real_output, *self._eager_outputs
        )

        self._outputs_tensor_tuple = convert_to_synced_tensor_tuple(
            self.__flatten_io("output", *self._eager_outputs)
        )
        self._eager_outputs_buffer = [
            self._eager_outputs,
        ]
        self._outputs_tensor_tuple_buffer = [
            self._outputs_tensor_tuple,
        ]

        # Make outputs buffer
        for i in range(self._outputs_buffer_size - 1):
            outputs_buffer_item, _ = self.__empty_like_io(
                "output", *self._eager_outputs
            )
            self._eager_outputs_buffer.append(outputs_buffer_item)
            outputs_tensor_tuple_buffer_item = convert_to_synced_tensor_tuple(
                self.__flatten_io("output", *outputs_buffer_item)
            )
            self._outputs_tensor_tuple_buffer.append(outputs_tensor_tuple_buffer_item)
        self.__check_outputs_buffer()

    def __check_outputs_buffer(self):
        has_len = len(self._outputs_tensor_tuple_buffer)
        assert (
            has_len == self._outputs_buffer_size
        ), f"nn.Graph's outputs buffer size {has_len} donot match the set value {self._outputs_buffer_size}."
        # Check there is not duplicated outputs buffer tensor.
        out_id_dic = dict()

        def check_id_and_add(t, name):
            if t is not None:
                tid = id(t)
                assert (
                    tid not in out_id_dic
                ), f"nn.Graph's outputs buffer add buffer tensor tid {tid} has conflict, new item name {name}, old item name {out_id_dic[tid]}."
                out_id_dic[tid] = name

        for b_idx, buffer in enumerate(self._outputs_tensor_tuple_buffer):
            for i_idx, item in enumerate(buffer):
                check_id_and_add(
                    item, "graph_ouputs_buffer_" + str(b_idx) + "_" + str(i_idx)
                )

    def __run(self, *args, **kwargs):
        self.__ensure_input_tensors_contiguous(*args, **kwargs)
        try:
            flattened_eager_args = self.__flatten_io("input", *args, **kwargs)
            outputs_tensor_tuple = self._outputs_tensor_tuple_buffer[
                self._cur_index_of_ouputs_buffer
            ]
            eager_outputs = self._eager_outputs_buffer[self._cur_index_of_ouputs_buffer]

            # oneflow._oneflow_internal.eager.Sync() NOTE(chengcheng): Need Sync?
            oneflow._oneflow_internal.nn.graph.RunLazyNNGraph(
                convert_to_tensor_tuple(flattened_eager_args),
                outputs_tensor_tuple,
                self._state_tensor_tuple,
                self._c_nn_graph,
            )
            # Update outputs buffer reading index
            self._cur_index_of_ouputs_buffer += 1
            if self._cur_index_of_ouputs_buffer >= self._outputs_buffer_size:
                self._cur_index_of_ouputs_buffer = 0
        except:
            self.__print(
                2,
                0,
                "[ERROR]"
                + self._shallow_repr()
                + " run got error: "
                + sys_exc_error_msg(),
            )
            raise

        # Copy outputs from buffer
        eager_outputs, _ = self.__copy_io("output", *eager_outputs)

        # Make sure that last used devices of tensors in `outputs_tensor_tuple` are
        # "critical_section".
        # NNGraph's execution flow will be broken if `last_used_device` of `outputs_tensor_tuple`
        # are not "critical_section".
        oneflow._oneflow_internal.nn.graph.SoftSyncNNGraphBuffers(
            outputs_tensor_tuple, self._c_nn_graph
        )
        # Always pack outputs to remain type of outputs
        return seq_to_func_return(eager_outputs, True)

    def __build_io(self, io_type, build_func, *args, **kwargs):
        assert io_type in ("input", "output")
        op_names = []
        args_repr = []
        tensor2op_name = {}

        def build_tensor_or_none(tensor, name, repr_str):
            assert tensor is None or (isinstance(tensor, Tensor))
            if isinstance(tensor, Tensor):
                build_arg = build_func(name, tensor)
                op_names.append(name)
                tensor2op_name[build_arg] = name
            else:
                build_arg = None

            args_repr.append(repr_str)
            self.__print(0, 1, repr_str)
            return build_arg

        args_tree = ArgsTree(
            (args, kwargs), True, "_" + self.name + "_" + io_type, None
        )

        def leaf_arg_fn(arg):
            name = arg.prefix() + "_" + arg.name()
            if isinstance(arg.value(), Tensor):
                arg_repr = self.__io_item_check_and_gen_repr(
                    arg.value(), Tensor, io_type, name
                )
                build_arg = build_tensor_or_none(arg.value(), name, arg_repr)
                return build_arg
            elif arg.value() is None:
                arg_repr = self.__io_item_check_and_gen_repr(
                    arg.value(), None, io_type, name
                )
                build_arg = build_tensor_or_none(arg.value(), name, arg_repr)
            else:  # Opaque
                # Error
                arg_repr = self.__io_item_check_and_gen_repr(
                    arg.value(), None, io_type, name
                )

        out = args_tree.map_leaf(leaf_arg_fn)
        build_args = out[0]
        build_kwargs = out[1]

        return op_names, build_args, build_kwargs, args_repr, tensor2op_name

    def __io_item_check_and_gen_repr(self, item, expect_type, io_type, name):
        assert io_type in ("input", "output")
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
            return repr_str
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
            return repr_str
        else:
            repr_str = (
                "[ERROR](" + io_type.upper() + ":" + name + ":" + str(type(item)) + ")"
            )
            self.__print(2, 0, repr_str)
            raise NotImplementedError(
                "nn.Graph.build()'s input/output item only support types: Tensor/None."
            )

    def __map_io(self, io_type, func, *args, **kwargs):
        assert io_type in ("input", "output")

        def mapping_tensor_or_none(tensor):
            assert tensor is None or (isinstance(tensor, Tensor))
            if isinstance(tensor, Tensor):
                mapped_arg = func(tensor)
            else:
                mapped_arg = None
            return mapped_arg

        args_tree = ArgsTree(
            (args, kwargs), True, "_" + self.name + "_" + io_type, None
        )

        def leaf_arg_fn(arg):
            arg_value = arg.value()
            if isinstance(arg_value, Tensor) or arg_value is None:
                return mapping_tensor_or_none(arg_value)
            else:
                self.__io_item_check(
                    arg_value, None, io_type, arg.prefix() + "_" + arg.name(),
                )

        out = args_tree.map_leaf(leaf_arg_fn)
        mapped_args = out[0]
        mapped_kwargs = out[1]
        return mapped_args, mapped_kwargs

    def __flatten_io(self, io_type, *args, **kwargs):
        flattened_args = []
        args_tree = ArgsTree((args, kwargs), False)

        for arg in args_tree.iter_nodes():
            if isinstance(arg, Tensor):
                flattened_args.append(arg)
            else:
                continue
        return flattened_args

    def __io_item_check(self, item, expect_type, io_type, name):
        if expect_type is None and item is None:
            return
        elif expect_type is not None and isinstance(item, expect_type):
            return
        else:
            assert io_type in ("input", "output")
            repr_str = (
                "[ERROR](" + io_type.upper() + ":" + name + ":" + str(type(item)) + ")"
            )
            self.__print(2, 0, repr_str)
            raise NotImplementedError(
                "nn.Graph.build()'s input/output item only support types: Tensor/None."
            )

    def __empty_like_io(self, io_type, *args, **kwargs):
        def func(t):
            shape = t.shape
            dtype = t.dtype

            with oneflow._oneflow_internal.lazy_mode.guard(False):
                if t.is_global:
                    eager_out = oneflow.empty(
                        shape, dtype=dtype, placement=t.placement, sbp=t.sbp,
                    )
                else:
                    eager_out = oneflow.empty(shape, dtype=dtype, device=t.device)

            return eager_out

        return self.__map_io(io_type, func, *args, **kwargs)

    def __copy_io(self, io_type, *args, **kwargs):
        def func(tensor):
            with oneflow._oneflow_internal.lazy_mode.guard(False):
                build_arg = tensor.to(copy=True)
                return build_arg

        return self.__map_io(io_type, func, *args, **kwargs)

    def _add_block(self, name: str, module: Module = None) -> None:
        r"""Adds module to the graph as a block so that the module will
        be called in nn.Graph.build.

        Args:
            name (str): name of the child block. The child block can be accessed from this graph using the given name.
            module (Module): child module to be added to the graph.

        Just assign nn.Module in nn.Graph, _add_block will be called to add the
        module as a Block:

        For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> class LinearGraph(flow.nn.Graph):
            ...     def __init__(self):
            ...         super().__init__()
            ...         # add a nn.Module as a block to graph.
            ...         self.linear = flow.nn.Linear(3, 8, False)
            ...     def build(self, x):
            ...         # call the nn.Module block.
            ...         return self.linear(x)


        The block can be accessed as an attribute using the given name.
            >>> g = LinearGraph()
            >>> print(repr(g.linear))
            (MODULE:linear:Linear(in_features=3, out_features=8, bias=False)): (
              (PARAMETER:linear.weight:tensor(..., size=(8, 3), dtype=oneflow.float32, requires_grad=True)): ()
            )
        """
        if "_name" not in self.__dict__:
            raise AttributeError(
                "Base class nn.Graph has not been initialized, "
                "please call super().__init__() in subclass of nn.Graph "
                "before assigning any attribute."
            )
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

        self._blocks[name] = get_block_cls(module)(
            "", name, module, weakref.proxy(self)
        )

    def __setattr__(self, name: str, value=None):
        if isinstance(value, Module):
            self._add_block(name, value)
        elif isinstance(value, Optimizer):
            raise AttributeError(
                "'{}' nn.Graph is not allowed to set Optimizer attribute named '{}'. "
                "Please use add_optimizer(...) instead.".format(
                    type(self).__name__, name
                )
            )
        elif isinstance(value, Tensor):
            raise AttributeError(
                "'{}' nn.Graph is not allowed to set Tensor attribute named '{}'. "
                "Please use nn.Module to hold the tensor, then add the nn.Module to nn.Graph.".format(
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

    def __del__(self):
        # Ensure vm has finished running this graph.
        if self._session._env.is_shutting_down():
            # After python shutting down, it's not safe to call oneflow._oneflow_internal.eager.
            # But shutting down will do sync in SwitchToShuttingDownPhase.
            # So it's safe to skip sync here.
            return
        oneflow._oneflow_internal.eager.Sync()
        current_env_enable_mlir_inference_opt = os.getenv(
            "ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"
        )
        if (self.env_enable_mlir_inference_opt is not None) and (
            current_env_enable_mlir_inference_opt is None
        ):
            os.environ[
                "ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"
            ] = self.env_enable_mlir_inference_opt
        oneflow._oneflow_internal.ClearVariableTensorMgr()

    def __ensure_input_tensors_contiguous(self, *args, **kwargs):
        args_tree = ArgsTree((args, kwargs), False)

        def func(value):
            if isinstance(value, Tensor) and not value.is_contiguous():
                value.contiguous_()
            return value

        args_tree.map_leaf(func)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
