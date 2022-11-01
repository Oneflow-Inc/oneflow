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
from typing import Iterator, Optional, Set, Union, List
import weakref

import oneflow._C
import oneflow._oneflow_internal
from oneflow.framework import graph_build_util
from oneflow.env import get_rank
from oneflow.framework.tensor import Tensor, TensorTuple
from oneflow.nn.module import Module
from oneflow.nn.modules.container import *
from oneflow.nn.utils.container import *
from oneflow.nn.parameter import Parameter
from oneflow.nn.graph.block_config import GraphBlockConfig
from oneflow.nn.graph.util import (
    add_indent,
    ArgsTree,
    operators_repr,
    seq_to_func_return,
)


def get_block_cls(item):
    if isinstance(item, Sequential):
        return GraphSequential
    elif isinstance(item, ModuleList):
        return GraphModuleList
    elif isinstance(item, ModuleDict):
        return GraphModuleDict
    elif isinstance(item, ParameterList):
        return GraphParameterList
    elif isinstance(item, ParameterDict):
        return GraphParameterDict
    elif isinstance(item, Module):
        return GraphModule
    elif isinstance(item, Tensor):
        return GraphTensor
    else:
        raise NotImplementedError()


class BlockType:
    NONE = "NONE"
    MODULE = "MODULE"
    PARAMETER = "PARAMETER"
    BUFFER = "BUFFER"


class GraphBlock(object):
    def __init__(
        self,
        prefix: str = "",
        name: str = "",
        belonged_graph: weakref.ProxyTypes = None,
    ):
        self._gname = name
        self._gname_prefix = prefix
        self._gtype = BlockType.NONE
        self._gorigin = None
        self._gscope = None
        self._gprev_scope = None
        assert belonged_graph is None or isinstance(belonged_graph, weakref.ProxyTypes)
        self._gbelonged_graph = belonged_graph
        self.nngraph_config = GraphBlockConfig()

    @property
    def nngraph_name(self):
        return self._gname

    @property
    def gname_prefix(self):
        return self._gname_prefix

    @property
    def gtype(self):
        return self._gtype

    @property
    def gprev_scope(self):
        if self._gprev_scope is None:
            self._gprev_scope = oneflow._oneflow_internal.GetCurrentScope()
        return self._gprev_scope

    @property
    def gscope(self):
        if self._gscope is None:
            self._gscope = graph_build_util.make_new_block_scope(self.gprev_scope, self)
        return self._gscope

    def gscope_context(self):
        return graph_build_util.BlockScopeContext(self.gprev_scope, self.gscope)


class GraphModule(GraphBlock):
    def __init__(
        self,
        prefix: str = "",
        name: str = "",
        origin: Module = None,
        belonged_graph: weakref.ProxyTypes = None,
    ):
        assert not isinstance(origin, GraphBlock)
        super().__init__(prefix, name, belonged_graph)
        self._gdebug = False
        self._gdebug_min_s_level = 2
        self._gdebug_max_v_level = 0
        self._gdebug_max_py_stack_depth = 2
        self._gdebug_only_user_py_stack = True
        self._gdebug_op_repr_with_py_stack = False
        self._gtype = BlockType.MODULE
        self._gis_executing_forward = False
        self._gmodules = OrderedDict()
        self._gparameters = OrderedDict()
        self._gbuffers = OrderedDict()
        self._gargs_repr = []
        self._gouts_repr = []
        self.gset_origin(origin)

    @property
    def nngraph_origin(self):
        return self._gorigin

    def gset_origin(self, origin):
        self._gorigin = origin
        if origin is None:
            return
        assert isinstance(origin, Module)
        for (n, m) in list(origin.named_children()):
            self.__setattr__(
                n,
                get_block_cls(m)(
                    self._gname_prefix + self._gname + ".", n, m, self._gbelonged_graph
                ),
            )
        for (n, p) in list(origin.named_parameters("", False)):
            self.__setattr__(
                n, get_block_cls(p)(self._gname_prefix + self._gname + ".", n, p)
            )
        for (n, b) in list(origin.named_buffers("", False)):
            self.__setattr__(
                n, get_block_cls(b)(self._gname_prefix + self._gname + ".", n, b)
            )

    def gdebug(
        self,
        v_level: int = 0,
        *,
        ranks: Optional[Union[int, List[int]]] = None,
        max_py_stack_depth: int = 2,
        only_user_py_stack=True,
        op_repr_with_py_stack=False,
    ) -> None:
        assert isinstance(v_level, int)
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

            self._debug_max_py_stack_depth = max_py_stack_depth
            self._debug_only_user_py_stack = only_user_py_stack
            self._debug_op_repr_with_py_stack = op_repr_with_py_stack

            if self._type == BlockType.MODULE:

                def _set_child(d):
                    for (_, n) in d.items():
                        n.debug(
                            v_level,
                            ranks=ranks,
                            max_py_stack_depth=max_py_stack_depth,
                            only_user_py_stack=only_user_py_stack,
                            op_repr_with_py_stack=op_repr_with_py_stack,
                        )

                _set_child(self._modules)

    def __call__(self, *args, **kwargs):
        assert self._type == BlockType.MODULE
        self.__print(0, 1, self._shallow_repr())

        args_tree = ArgsTree(
            (args, kwargs), True, "_" + self.name_prefix + self.name + "_input", None
        )

        for (name, arg) in args_tree.iter_named_nodes():
            if arg.is_leaf():
                arg_value = arg.value()
                meta_repr_str = (
                    arg_value._meta_repr()
                    if isinstance(arg_value, Tensor)
                    else str(type(arg_value))
                )
                in_str = "(INPUT:" + name + ":" + meta_repr_str + ")"
                if not isinstance(arg_value, Tensor):
                    in_str = "[WARNING]" + in_str
                self._args_repr.append(in_str)
                self.__print(0, 1, in_str)

        def _print_state(d):
            for (_, n) in d.items():
                self.__print(0, 1, n._shallow_repr())

        _print_state(self._parameters)
        _print_state(self._buffers)

        # NOTE: The original nn.Moudle's __call__ method is ignored, which means
        # that hooks of nn.Modules are ignored. It is not recommended
        # to use hooks of nn.Module in nn.Graph for the moment.
        # result = self._origin.__class__.__call__(self, *args)
        with graph_build_util.DebugScopeContext(
            self._debug_min_s_level,
            self._debug_max_v_level,
            self._debug,
            self._debug_max_py_stack_depth,
            self._debug_only_user_py_stack,
        ):
            result = self.__block_forward(*args, **kwargs)

        outputs = ()
        if not (type(result) is tuple or type(result) is list):
            outputs = (result,)
        else:
            outputs = result

        args_tree = ArgsTree(
            (outputs, {}), True, "_" + self.name_prefix + self.name + "_output", None
        )

        for (name, arg) in args_tree.iter_named_nodes():
            if arg.is_leaf():
                arg_value = arg.value()
                meta_repr_str = (
                    arg_value._meta_repr()
                    if isinstance(arg_value, Tensor)
                    else str(type(arg_value))
                )
                out_str = "(OUTPUT:" + name + ":" + meta_repr_str + ")"
                if not isinstance(arg_value, Tensor):
                    out_str = "[WARNING]" + out_str
                self._outs_repr.append(out_str)
                self.__print(0, 1, out_str)

        return result

    def __block_forward(self, *args, **kwargs):
        self._is_executing_forward = True
        args, kwargs = self.__pre_forward_map(*args, **kwargs)
        with self.scope_context():
            # "Instance method __func__ is the function object", "when an instance method object is called,
            # the underlying function __func__ is called, inserting the class instance __self__ in front of
            # the argument list."
            # Reference: https://docs.python.org/3/reference/datamodel.html
            unbound_forward_of_module_instance = self._origin.forward.__func__
            result = unbound_forward_of_module_instance(self, *args, **kwargs)
        self._is_executing_forward = False
        return result

    def __pre_forward_map(self, *args, **kwargs):
        # Insert identity op when doing activation checkpointing or pipeline execution.
        # Identity op outside activation checkpointing scope will be the endpoint of an activation checkpointing segment.
        # Identity op as the first op of a pipeline stage will make backward op depends on the identity op within the stage,
        # otherwise the backward op may depends the op in former stage which will make graph creates unnessary buffers.
        if self.config._stage_placement is not None:

            def insert_to_global(t):
                assert isinstance(t, Tensor)
                return self.__get_or_create_global(t, self.config._stage_placement)

            args, kwargs = self.__map_io(
                "input", insert_to_global, "insert_to_global", *args, **kwargs
            )

        if self.config.activation_checkpointing or (
            self.config.stage_id is not None and self.config.stage_id >= 0
        ):

            def insert_identity(t):
                assert isinstance(t, Tensor)
                return self.__get_or_create_identity(t)

            args, kwargs = self.__map_io(
                "input", insert_identity, "insert_identity", *args, **kwargs
            )

        return args, kwargs

    def __get_or_create_global(self, input_tensor: Tensor = None, placement=None):
        assert input_tensor is not None
        assert placement is not None
        key = str(id(input_tensor)) + str(placement)

        # input_tensor + placement -> unique_global_tensor
        if key not in self._gbelonged_graph._unique_global_op_dict:
            # store input tensor to avoid tensor id recycle
            self._gbelonged_graph._unique_global_op_dict[key] = (
                input_tensor.to_global(placement=placement),
                input_tensor,
            )

        return self._gbelonged_graph._unique_global_op_dict[key][0]

    def __get_or_create_identity(self, input_tensor: Tensor = None):
        assert input_tensor is not None
        key = input_tensor

        # input_tensor(with placement) -> unique_identity_tensor
        # When placement is different, the input tensor(output tensor of __get_or_create_global) is different, so the
        # key can use only input tensor.
        if key not in self._gbelonged_graph._unique_identity_op_dict:
            # Reuse current module name for indentity op
            ident_name_scope = graph_build_util.make_new_name_scope(
                self.gprev_scope, self.name_prefix + self.name
            )
            with graph_build_util.BlockScopeContext(self.gprev_scope, ident_name_scope):
                # store input tensor to avoid tensor id recycle
                self._gbelonged_graph._unique_identity_op_dict[
                    key
                ] = oneflow._C.identity(input_tensor)

        return self._gbelonged_graph._unique_identity_op_dict[key]

    def add_module(self, name: str, module: Optional[Module]) -> None:
        self.__setattr__(
            name,
            get_block_cls(module)(
                self._gname_prefix + self._gname + ".", name, module, self._gbelonged_graph
            ),
        )

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        self.__setattr__(
            name,
            get_block_cls(param)(self._gname_prefix + self._gname + ".", name, param),
        )

    def modules(self, memo: Optional[Set["GraphBlock"]] = None) -> Iterator["GraphBlock"]:
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

    def __map_io(self, io_type, func, func_desc, *args, **kwargs):
        assert isinstance(func_desc, str)
        assert io_type in ("input", "output")
        mapped_args = []

        def map_tensor(item):
            assert isinstance(item, Tensor)
            return func(item)

        args_tree = ArgsTree(
            (args, kwargs),
            True,
            "_" + self.name_prefix + self.name + "_" + io_type,
            None,
        )

        def leaf_node_fn(leaf_node):
            arg = leaf_node.value()
            name = leaf_node.prefix() + "_" + leaf_node.name()
            is_tensor, repr_str = self.__io_tensor_check_and_gen(arg, io_type, name)
            if is_tensor:
                self.__print(
                    0,
                    1,
                    f"{repr_str} is a Tensor, {func_desc} transformation has been done.",
                )
                return map_tensor(arg)
            else:
                self.__print(
                    0,
                    0,
                    f"{repr_str} is not a Tensor, {func_desc} transformation will be ignored.",
                )
                return arg

        out = args_tree.map_leaf(leaf_node_fn)
        mapped_args = out[0]
        mapped_kwargs = out[1]
        return mapped_args, mapped_kwargs

    def __io_tensor_check_and_gen(self, item, io_type, name):
        assert io_type in ("input", "output")
        if isinstance(item, Tensor):
            repr_str = (
                "(" + io_type.upper() + ":" + name + ":" + item._meta_repr() + ")"
            )
            return True, repr_str
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
            return False, repr_str

    def __members(self, get_members_fn, recurse=True) -> Iterator["GraphBlock"]:
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

    def parameters(self, recurse: bool = True) -> Iterator["GraphBlock"]:
        assert self._type == BlockType.MODULE
        gen = self.__members(lambda module: module._parameters.items(), recurse=recurse)
        for elem in gen:
            yield elem

    def buffers(self, recurse: bool = True) -> Iterator["GraphBlock"]:
        assert self._type == BlockType.MODULE
        gen = self.__members(lambda module: module._buffers.items(), recurse=recurse)
        for elem in gen:
            yield elem

    def __setattr__(self, name: str, value=None) -> None:
        if value is None or not isinstance(value, GraphBlock):
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
                            self._gname, name
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
        # support get module
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        # support get parameter
        p_state = self._get_from_states(name, "_parameters")
        if p_state is not None:
            return p_state
        # support get buffer
        b_state = self._get_from_states(name, "_buffers")
        if b_state is not None:
            return b_state
        # support none parameter or buffer
        if name in self._origin._parameters:
            p_none = self._origin._parameters[name]
            assert p_none is None
            return None
        if name in self._origin._buffers:
            b_none = self._origin._buffers[name]
            assert b_none is None
            return None
        # support get normal attr
        if name in self._origin.__dict__:
            return self._origin.__dict__[name]
        # support get function
        if hasattr(self._origin, name):
            return partial(getattr(self._origin.__class__, name), self)
        raise AttributeError(
            "'{}' '{}' object '{}' in nn.Graph has no attribute '{}'".format(
                self._type, type(self).__name__, self._gname_prefix + self.name, name
            )
        )

    def _get_from_states(self, name, states_name):
        if states_name not in self.__dict__:
            return None

        _states = self.__dict__[states_name]
        if name not in _states:
            return None

        _s_block = _states[name]
        if graph_build_util.lazy_mode.is_enabled():
            _s_block.try_build()
            return _s_block.lazy_origin
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
        child_lines = []
        if (self.config is not None) and (not self.config._is_null):
            child_lines.append(add_indent(repr(self.config), 2))
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

        for op_str in self._ops_repr():
            child_lines.append(add_indent(op_str, 2))

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
            + self._gname_prefix
            + self._gname
            + ":"
            + self._origin._shallow_repr()
            + ")"
        )
        return shallow_repr

    def _ops_repr(self):
        r"""Generate operators' string representation of this module
        """
        assert self._gbelonged_graph, (
            "GraphModule: "
            + self._gname_prefix
            + self.name
            + "'s belonged graph is not set."
        )

        if self._gbelonged_graph.is_compiled:
            if self._gbelonged_graph._compiled_graph_proto is not None:
                module_conf = self._gbelonged_graph._compiled_graph_proto.module_name2module_conf[
                    self.name_prefix + self.name
                ]
                return operators_repr(
                    module_conf.ops,
                    self._gbelonged_graph._compiled_graph_proto,
                    self._debug_op_repr_with_py_stack,
                )

        return []

    def __print(self, s_level=2, v_level=0, msg: str = ""):
        r"""Do print according to info level.
        """
        assert isinstance(s_level, int)
        assert isinstance(v_level, int)
        assert isinstance(msg, str)
        if s_level >= self._debug_min_s_level:
            if (s_level > 0) or (s_level == 0 and v_level <= self._debug_max_v_level):
                print(msg, flush=True)


class LazyBuilder(object):
    def __init__(self, name: str = None, method=None):
        self.name = name
        self.method = method
        self.result = None
        self.finished = False

    def try_build(self, block=None):
        if not self.finished:
            assert self.name is not None
            assert self.method is not None
            assert self.result is None
            with block.gscope_context():
                self.result = self.method()
            self.finished = True


class GraphTensor(GraphBlock):
    def __init__(
        self,
        prefix: str = "",
        name: str = "",
        origin: Union[Parameter, Tensor] = None,
        belonged_graph: weakref.ProxyTypes = None,
    ):
        assert not isinstance(origin, GraphBlock)
        super().__init__(prefix, name, belonged_graph)
        if isinstance(origin, Parameter):
            self._type = BlockType.PARAMETER
        elif isinstance(origin, Tensor):
            self._type = BlockType.BUFFER
        else:
            raise NotImplementedError()
        self._lazy_origin_builder = LazyBuilder()
        self.build_finished = False
        self.set_origin(origin)

    @property
    def gorigin(self):
        return self._origin

    def gset_origin(self, origin):
        self._origin = origin

    @property
    def glazy_origin(self):
        assert (
            self._type == BlockType.PARAMETER or self._type == BlockType.BUFFER
        ), "Only Parameter or Buffer GraphBlock has lazy_origin"
        return self._lazy_origin_builder.result

    def glazy_origin_builder(self):
        assert (
            self._type == BlockType.PARAMETER or self._type == BlockType.BUFFER
        ), "Only Parameter or Buffer GraphBlock has lazy_origin_builder"
        return self._lazy_origin_builder

    def gset_lazy_origin_builder(self, builder=None):
        assert (
            self._type == BlockType.PARAMETER or self._type == BlockType.BUFFER
        ), "Only Parameter or Buffer GraphBlock has lazy_origin_builder"
        self._lazy_origin_builder = builder

    def gtry_build(self):
        if not self.build_finished:
            self._lazy_origin_builder.try_build(self)
            self.build_finished = True

    def __repr__(self):
        lines = None
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
            + self._gname_prefix
            + self._gname
            + ":"
            + self._origin._meta_repr()
            + ")"
        )
        return shallow_repr


class GraphSequential(get_seq(GraphModule)):
    def __init__(
        self,
        prefix: str = "",
        name: str = "",
        origin: Sequential = None,
        belonged_graph: weakref.ProxyTypes = None,
    ):
        super().__init__()
        self._gname_prefix = prefix
        self._gname = name
        self._gbelonged_graph = belonged_graph
        self.set_origin(origin)


class GraphModuleList(get_list(GraphModule)):
    def __init__(
        self,
        prefix: str = "",
        name: str = "",
        origin: ModuleList = None,
        belonged_graph: weakref.ProxyTypes = None,
    ):
        super().__init__()
        self._gname_prefix = prefix
        self._gname = name
        self._gbelonged_graph = belonged_graph
        self.set_origin(origin)
        # MoudleList is a container without forward() method,
        # so it will not be executed or has an execution config.
        self.config = None


class GraphModuleDict(get_dict(GraphModule)):
    def __init__(
        self,
        prefix: str = "",
        name: str = "",
        origin: ModuleDict = None,
        belonged_graph: weakref.ProxyTypes = None,
    ):
        super().__init__()
        self._gname_prefix = prefix
        self._gname = name
        self._gbelonged_graph = belonged_graph
        self.set_origin(origin)


class GraphParameterList(get_para_list(GraphModule)):
    def __init__(
        self,
        prefix: str = "",
        name: str = "",
        origin: ParameterList = None,
        belonged_graph: weakref.ProxyTypes = None,
    ):
        super().__init__()
        self._gname_prefix = prefix
        self._gname = name
        self._gbelonged_graph = belonged_graph
        self.set_origin(origin)
        self._is_executing_forward = True

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx = self._get_abs_string_index(idx)
        key = str(idx)
        p_state = self._get_from_states(key, "_parameters")
        if p_state is not None:
            return p_state
        else:
            raise AttributeError("ParameterList dosen't contain ", key)


class GraphParameterDict(get_para_dict(GraphModule)):
    def __init__(
        self,
        prefix: str = "",
        name: str = "",
        origin: ParameterDict = None,
        belonged_graph: weakref.ProxyTypes = None,
    ):
        super().__init__()
        self._gname_prefix = prefix
        self._gname = name
        self._gbelonged_graph = belonged_graph
        self.set_origin(origin)
        self._is_executing_forward = True

    def __getitem__(self, key: str):
        p_state = self._get_from_states(key, "_parameters")
        if p_state is not None:
            return p_state
        else:
            raise AttributeError("ParameterDict dosen't contain key ", key)
