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
import os
import time
from collections import OrderedDict
from functools import partial
from typing import Dict, Optional, Union, List
from google.protobuf import text_format

import oneflow
import oneflow._oneflow_internal
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
from oneflow.nn.graph.util import add_indent, seq_to_func_return, sys_exc_error_msg
from oneflow.nn.module import Module
from oneflow.nn.optimizer.lr_scheduler import LrScheduler
from oneflow.nn.optimizer.optimizer import Optimizer
from oneflow.nn.optimizer.sparse_optimizer import SparseOptimizer


class Graph(object):
    r"""Base class for training or evaluating a neural network in graph mode.

    To use graph mode for model training or evaluation in OneFlow, you should:

    1. Define your customized graph as a subclass of ``nn.Graph``.
    2. Add ``super().__init__()`` in your subclass's ``__init__()``.
    3. Add modules to your graph as regular attributes.
    4. Define computation logical in ``build()`` method.
    5. Instantiate your graph then call it.

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

    Note that Graph cannot be nested at the moment.
    """
    _child_init_cnt = dict()

    def __init__(self):
        """
        Initializes internal Graph states. It MUST be called in ``__init__`` method of subclass.

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
        self._grad_scaler = None
        self._variables_conf = OrderedDict()
        self._is_compiled = False
        # forward graph job proto
        self._forward_job_proto = None
        # forward, backward and optimized graph job proto
        self._full_job_proto = None
        self._args_repr = []
        self._outs_repr = []
        self._debug = False
        self._debug_min_s_level = 2
        self._debug_max_v_level = 0
        self._outputs_buffer_size = 2
        self._cur_index_of_ouputs_buffer = 0

        self._c_nn_graph = oneflow._oneflow_internal.nn.graph.CNNGraph(self._name)
        session = session_ctx.GetDefaultSession()
        assert type(session) is MultiClientSession
        session.TryInit()
        session.AddCGraph(self._c_nn_graph)

    def build(self, *args):
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

        .. code-block:: python

            >>> import oneflow as flow
            >>> class MyGraph(flow.nn.Graph):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = flow.nn.Linear(3, 8, False)
            ...     def build(self, x):
            ...         return self.linear(x)

            >>> linear_graph = MyGraph()
            >>> x = flow.randn(4, 3)
            >>> y = linear_graph(x) # The build() method is called implicitly

        Note that ``build()`` method's inputs and outputs only accept positional
        arguements at the moment, each argument must be one of these types:

        * ``Tensor``
        * ``list`` of ``Tensor``
        * ``None``

        """
        raise NotImplementedError()

    def add_optimizer(
        self, optim: Optimizer, *, lr_sch: LrScheduler = None,
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
        in ``nn.Graph.build()`` for the moment. So you may call ``Tensor.sum()``
        or ``Tensor.mean()`` to make the loss tensor a scalar tensor.

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
            >>> for t in range(3):
            ...     loss = linear_graph(x, y)

        Args:
            optim (oneflow.optim.Optimizer): The optimizer.
            lr_sch : The learning rate scheduler, see oneflow.optim.lr_scheduler.
        """
        opt_dict = dict()
        assert optim is not None, "optimizer cannot be None"
        assert isinstance(
            optim, (Optimizer, SparseOptimizer)
        ), "optimizer must be an instance of Optimizer"
        opt_dict["optim"] = optim
        if lr_sch is not None:
            assert isinstance(lr_sch, LrScheduler)
            assert (
                lr_sch._optimizer is optim
            ), "lr_scheduler's optimizer must be the same optimizer in add_optimizer."
            opt_dict["lr_sch"] = lr_sch
        self._opts.append(opt_dict)
        # Set the training config if there is an optimizer add in graph.
        if len(self._opts) == 1:
            self.config._train(True)

    def set_grad_scaler(self, grad_scaler: GradScaler = None):
        r"""Set the GradScaler for gradient and loss scaling.
        """
        assert isinstance(grad_scaler, (GradScaler, StaticGradScaler))
        self._grad_scaler = grad_scaler

    def __call__(self, *args):
        r"""Call nn.Graph subclass instance to run your customized graph.

        Call your customized graph after the instantiation:

        .. code-block:: python

            g = CustomGraph()
            out_tensors = g(input_tensors)

        The inputs of ``__call__`` method must match the inputs of ``build()``
        method. And the ``__call__`` method will return outputs matching the
        outputs of ``build()`` method.

        Note that the first call takes longer than later calls, because nn.Graph
        will do the computaion graph generation and optimization at the first call.

        Donot override this function.
        """
        if not self._is_compiled:
            self._compile(*args)

        return self._run(*args)

    @property
    def name(self):
        r"""Name auto-generated for this graph.
        """
        return self._name

    @property
    def training(self):
        r"""In traninig mode if the graph has an optimizer.
        """
        return self.config.training

    def debug(
        self,
        v_level: int = 0,
        ranks: Optional[Union[int, List[int]]] = None,
        mode: bool = True,
    ) -> None:
        r"""Open or close debug mode of the graph.

        If in debug mode, logs of computation graph building infos or warnings will be
        printed. Otherwise, only errors will be printed.

        Use ``v_level`` to choose verbose debug info level, default level is 0, max level is 1.
        ``v_level`` 0 will print warning and graph creating stages. ``v_level`` 1 will additionally
        print graph build info of each module.
        
        Use ``ranks`` to choose which rank to print the debug information.

        .. code-block:: python

            g = CustomGraph()
            g.debug()  # Open debug mode
            out_tensors = g(input_tensors)  # Will print log for debug at the first call

        Args:
            v_level (int): choose verbose debug info level, default v_level is 0, max v_level is 1.
            ranks (int or list(int)): choose ranks to print the debug information. Default rank ``0``.
                You can choose any valid rank. Ranks equals ``-1`` means debug on all ranks.
            mode (bool): whether to set debug mode (``True``) or not (``False``). Default: ``True``.
        """
        assert isinstance(v_level, int)
        assert isinstance(mode, bool)

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
            self._debug = mode
            if self._debug:
                self._debug_min_s_level = 0
                self._debug_max_v_level = v_level
            for name, block in self._blocks.items():
                assert block.type == BlockType.MODULE
                block.debug(v_level, ranks, mode)

    def __repr__(self):
        r"""For printing the graph structure.

        The graph structure can be printed after graph instantiation.

        After the first call of graph, inputs and outputs will be added to
        the graph structure.

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

    def _print(self, s_level=2, v_level=0, msg: str = ""):
        r"""Do print according to info level.
        """
        assert isinstance(s_level, int)
        assert isinstance(v_level, int)
        assert isinstance(msg, str)
        if s_level >= self._debug_min_s_level:
            if (s_level > 0) or (s_level == 0 and v_level <= self._debug_max_v_level):
                print(msg)

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
        if not self._is_compiled:
            self._print(
                2,
                0,
                f"[ERROR]{self._shallow_repr()} has not been compiled, so it's graph proto is None."
                " You can call the graph to trigger it's compilation.",
            )
        return self._forward_job_proto

    @property
    def _full_graph_proto(self):
        if not self._is_compiled:
            self._print(
                2,
                0,
                f"[ERROR]{self._shallow_repr()} has not been compiled, so it's full graph proto is None."
                " You can call the graph to trigger it's compilation.",
            )
        return self._full_job_proto

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

    def _filter_states(self):
        state_tensor_set = set()
        state_tensors = []
        state_op_names = []

        for state_block in self._state():
            state_tensor = state_block.origin
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
        self.config.proto.set_job_name(self._name)
        self._outputs_buffer_size = self.config._outputs_buffer_size

        if self._grad_scaler is not None:
            self._grad_scaler._generate_conf_for_graph(
                self.config.proto.mutable_train_conf()
            )

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

    def _compile(self, *args):
        # Build graph
        try:
            self._print(0, 0, self._shallow_repr() + " start building graph.")
            assert not self._is_compiled, (
                "nn.Graph " + self._name + " has already been compiled."
            )
            build_graph_start = time.perf_counter()
            eager_outputs = self._build_graph(*args)
            build_graph_end = time.perf_counter()
            self._print(
                0,
                0,
                self._shallow_repr()
                + " building graph Done! Cost time: "
                + str(round(build_graph_end - build_graph_start, 2))
                + "s."
                + "\n",
            )
        except:
            self._print(
                2,
                0,
                "[ERROR]"
                + self._shallow_repr()
                + " building graph got error: "
                + sys_exc_error_msg(),
            )
            raise

        # Complie graph to execution plan and init Runtime
        try:
            self._print(
                0, 0, self._shallow_repr() + " start building plan.",
            )
            compile_and_init_start = time.perf_counter()
            self._c_nn_graph.complie_and_init_runtime()
            compile_and_init_end = time.perf_counter()
            self._print(
                0,
                0,
                self._shallow_repr()
                + " building plan Done! Cost time: "
                + str(round(compile_and_init_end - compile_and_init_start, 2))
                + "s."
                + "\n"
                + self._shallow_repr()
                + "'s total time to build graph and plan : "
                + str(round(compile_and_init_end - build_graph_start, 2))
                + "s."
                + "\n",
            )
        except:
            self._print(
                2,
                0,
                "[ERROR]"
                + self._shallow_repr()
                + " building plan got error: "
                + sys_exc_error_msg(),
            )
            raise

        self._is_compiled = True
        return eager_outputs

    def _build_graph(self, *args):
        session = session_ctx.GetDefaultSession()
        assert type(session) is MultiClientSession

        # Filter to get unique states in graph
        state_op_names = self._filter_states()

        self._generate_config_proto()

        # Deal with parameter and buffer
        self._print(
            0,
            1,
            self._shallow_repr()
            + " start building graph builders of parameters and buffers.",
        )
        self._create_states_builder()
        self._print(
            0,
            1,
            self._shallow_repr()
            + " end building graph builders of parameters and buffers.",
        )

        with graph_build_util.graph_build_context(self.config.proto, session):
            # Deal with inputs
            self._print(0, 1, self._shallow_repr() + " start building graph inputs.")
            arg_op_names, lazy_args, self._args_repr, _ = self._build_io(
                "input", graph_build_util.build_graph_input_arg, *args
            )
            self._print(0, 1, self._shallow_repr() + " end building graph inputs.")

            # Deal with module in self.build(*args)
            self._print(0, 1, self._shallow_repr() + " start building graph modules.")
            outputs = self.build(*lazy_args)
            self._print(0, 1, self._shallow_repr() + " end building graph modules.")

            # Deal with outputs
            self._print(0, 1, self._shallow_repr() + " start building graph outputs.")
            if not (type(outputs) is tuple or type(outputs) is list):
                if outputs is None:
                    outputs = ()
                else:
                    outputs = (outputs,)

            (
                output_op_names,
                self._eager_outputs,
                self._outs_repr,
                out2name,
            ) = self._build_io("output", graph_build_util.build_graph_output, *outputs)

            self._print(0, 1, self._shallow_repr() + " end building graph outputs.")

            # Save forward graph job proto
            self._forward_job_proto = c_api_util.GetCurrentJob()

            self._print(
                0,
                1,
                self._shallow_repr() + " start building graph with compile passes.",
            )
            # Complete the graph job proto
            oneflow._oneflow_internal.CurJobBuildAndInferCtx_Complete()
            # Save full graph job proto after job Complete for find real output blob shape and build it.
            self._full_job_proto = c_api_util.GetCurrentJob()
            self._print(
                0, 1, self._shallow_repr() + " end building graph with compile passes."
            )

            # Re-build outputs accoring to full graph and outputs buffer config.
            self._print(
                0,
                1,
                self._shallow_repr()
                + " start re-building graph outputs for optimizatioin.",
            )
            self._rebuild_outputs(out2name)
            self._print(
                0,
                1,
                self._shallow_repr()
                + " end re-building graph outputs for optimizatioin.",
            )

            # Register input/output/variable/buffer to _c_nn_graph
            self._c_nn_graph.register_input_op_names_and_tensors(
                arg_op_names, convert_to_tensor_tuple(self._flatten_io("input", *args))
            )
            self._c_nn_graph.register_output_op_names_and_tensors(
                output_op_names, self._outputs_tensor_tuple
            )
            self._c_nn_graph.register_variable_op_names_and_tensors(
                state_op_names, self._state_tensor_tuple
            )

        return seq_to_func_return(self._eager_outputs_buffer[0])

    def _rebuild_outputs(self, out2name=None):
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
                if fake_eager_out.is_consistent:
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

        self._eager_outputs = self._mapping_io(
            "output", build_real_output, *self._eager_outputs
        )

        self._outputs_tensor_tuple = convert_to_synced_tensor_tuple(
            self._flatten_io("output", *self._eager_outputs)
        )
        self._eager_outputs_buffer = [
            self._eager_outputs,
        ]
        self._outputs_tensor_tuple_buffer = [
            self._outputs_tensor_tuple,
        ]

        # Make outputs buffer
        for i in range(self._outputs_buffer_size - 1):
            outputs_buffer_item = self._empty_like_io("output", *self._eager_outputs)
            self._eager_outputs_buffer.append(outputs_buffer_item)
            outputs_tensor_tuple_buffer_item = convert_to_synced_tensor_tuple(
                self._flatten_io("output", *outputs_buffer_item)
            )
            self._outputs_tensor_tuple_buffer.append(outputs_tensor_tuple_buffer_item)
        self._check_outputs_buffer()

    def _check_outputs_buffer(self):
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

    def _run(self, *args):
        try:
            flattened_eager_args = self._flatten_io("input", *args)
            outputs_tensor_tuple = self._outputs_tensor_tuple_buffer[
                self._cur_index_of_ouputs_buffer
            ]
            eager_outputs = self._eager_outputs_buffer[self._cur_index_of_ouputs_buffer]

            # oneflow._oneflow_internal.eager.multi_client.Sync() NOTE(chengcheng): Need Sync?
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
            self._print(
                2,
                0,
                "[ERROR]"
                + self._shallow_repr()
                + " run got error: "
                + sys_exc_error_msg(),
            )
            raise

        # Copy outputs from buffer
        eager_outputs = self._copy_io("output", *eager_outputs)

        # Make sure that last used devices of tensors in `outputs_tensor_tuple` are
        # "critical_section".
        # NNGraph's execution flow will be broken if `last_used_device` of `outputs_tensor_tuple`
        # are not "critical_section".
        oneflow._oneflow_internal.nn.graph.SoftSyncNNGraphBuffers(
            outputs_tensor_tuple, self._c_nn_graph
        )
        return seq_to_func_return(eager_outputs)

    def _build_io(self, io_type, build_func, *args):
        assert io_type in ("input", "output")
        io_type_upper = io_type.upper()
        build_args = []
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
            self._print(0, 1, repr_str)
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

        return op_names, build_args, args_repr, tensor2op_name

    def _mapping_io(self, io_type, func, *args):
        assert io_type in ("input", "output")
        io_type_upper = io_type.upper()
        mapped_args = []

        def mapping_tensor_or_none(tensor):
            assert tensor is None or (isinstance(tensor, Tensor))
            if isinstance(tensor, Tensor):
                mapped_arg = func(tensor)
            else:
                mapped_arg = None
            return mapped_arg

        for idx, arg in enumerate(args):
            if isinstance(arg, Tensor) or arg is None:
                mapped_args.append(mapping_tensor_or_none(arg))
            elif isinstance(arg, (TensorTuple, list)):
                if isinstance(arg, TensorTuple):
                    seq_args = TensorTuple()
                else:
                    seq_args = list()
                for i in range(len(arg)):
                    seq_args.append(mapping_tensor_or_none(arg[i]))
                mapped_args.append(seq_args)
            else:
                self._io_item_check(arg, None, io_type, idx)

        return mapped_args

    def _empty_like_io(self, io_type, *args):
        def func(t):
            shape = t.shape
            dtype = t.dtype

            with oneflow._oneflow_internal.lazy_mode.guard(False):
                if t.is_consistent:
                    eager_out = oneflow.empty(
                        shape, dtype=dtype, placement=t.placement, sbp=t.sbp,
                    )
                else:
                    eager_out = oneflow.empty(shape, dtype=dtype, device=t.device)

            return eager_out

        return self._mapping_io(io_type, func, *args)

    def _copy_io(self, io_type, *args):
        def func(tensor):
            with oneflow._oneflow_internal.lazy_mode.guard(False):
                build_arg = tensor.to(copy=True)
                return build_arg

        return self._mapping_io(io_type, func, *args)

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
            self._print(2, 0, repr_str)
            raise NotImplementedError(
                "nn.Graph.build()'s input/output only support types: Tensor/list(Tensor)/None."
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
            self._print(2, 0, repr_str)
            raise NotImplementedError(
                "nn.Graph.build()'s input/output only support types: Tensor/list(Tensor)/None."
            )

    def _add_block(self, name: str, module: Module = None) -> None:
        r"""Adds module to the graph as a block so that the module will
        be called in nn.Graph.build.

        Args:
            name (str): name of the child block. The child block can be accessed from this graph using the given name.
            module (Module): child module to be added to the graph.

        Just assign nn.Module in nn.Graph, _add_block will be called to add the
        module as a Block:

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

        self._blocks[name] = get_block_cls(module)("", name, module)

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


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
