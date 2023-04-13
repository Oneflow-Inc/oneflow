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
from typing import Iterator, Optional, Set, Union, List
import weakref
import types

import oneflow._C
import oneflow._oneflow_internal
from oneflow.framework import graph_build_util
from oneflow.framework.tensor import Tensor, TensorTuple
from oneflow.nn.modules.module import Module
from oneflow.nn.modules.container import *
from oneflow.nn.utils.container import *
from oneflow.nn.parameter import Parameter
from oneflow.nn.graph.graph_block import (
    GraphBlockType,
    GraphBlock,
    GraphModule,
    GraphTensor,
)
from oneflow.nn.graph.util import (
    add_indent,
    seq_to_func_return,
)
from oneflow.framework.args_tree import ArgsTree


def get_proxy_cls(item):
    if isinstance(item, Sequential):
        return ProxySequential
    elif isinstance(item, ModuleList):
        return ProxyModuleList
    elif isinstance(item, ModuleDict):
        return ProxyModuleDict
    elif isinstance(item, ParameterList):
        return ProxyParameterList
    elif isinstance(item, ParameterDict):
        return ProxyParameterDict
    elif isinstance(item, Module):
        return ProxyModule
    elif isinstance(item, Tensor):
        return ProxyTensor
    else:
        raise NotImplementedError()


class Proxy(object):
    def __init__(self):
        """ An ecution proxy of nn.Module or Tensor.

        A proxy contains the original data(nn.Module or Tensor) and a graph representation of the original data.
        """
        # The original data
        self._oneflow_internal_origin__ = None
        # The graph representation of the original data
        self._oneflow_internal_graphblock__ = None

    def to(self, *args, **kwargs):
        """
        """
        if len(args) == 1 and issubclass(args[0], GraphBlock):
            return self._oneflow_internal_graphblock__
        elif len(args) == 1 and (args[0] is Module or args[0] is Tensor):
            return self._oneflow_internal_origin__
        else:
            self._oneflow_internal_origin__.to(*args, **kwargs)


class ProxyModule(Proxy):
    def __init__(
        self,
        origin: Module = None,
        prefix: str = "",
        name: str = "",
        belonged_graph: weakref.ProxyTypes = None,
    ):
        assert not isinstance(origin, Proxy)
        super().__init__()
        self._oneflow_internal_graphblock__ = GraphModule(
            prefix, name, belonged_graph, weakref.proxy(self)
        )
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()

        self._oneflow_internal_graphblock__set_origin(origin)

    def _oneflow_internal_graphblock__set_origin(self, origin):
        self._oneflow_internal_origin__ = origin
        if origin is None:
            return
        assert isinstance(origin, Module)
        for (n, m) in origin._modules.items():
            self.__setattr__(
                n,
                get_proxy_cls(m)(
                    m,
                    self.to(GraphModule)._name_prefix
                    + self.to(GraphModule)._name
                    + ".",
                    n,
                    self.to(GraphModule)._belonged_graph,
                ),
            )
        for (n, p) in list(origin.named_parameters("", False)):
            self.__setattr__(
                n,
                get_proxy_cls(p)(
                    p,
                    self.to(GraphTensor)._name_prefix
                    + self.to(GraphTensor)._name
                    + ".",
                    n,
                ),
            )
        for (n, b) in list(origin.named_buffers("", False)):
            self.__setattr__(
                n,
                get_proxy_cls(b)(
                    b,
                    self.to(GraphTensor)._name_prefix
                    + self.to(GraphTensor)._name
                    + ".",
                    n,
                ),
            )

    def __call__(self, *args, **kwargs):
        assert self.to(GraphModule)._type == GraphBlockType.MODULE
        self.__print(0, 1, self._shallow_repr())

        args_tree = ArgsTree(
            (args, kwargs),
            True,
            "_"
            + self.to(GraphModule).name_prefix
            + self.to(GraphModule).name
            + "_input",
            None,
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
                self.to(GraphModule)._args_repr.append(in_str)
                self.__print(0, 1, in_str)

        def _print_state(d):
            for (_, n) in d.items():
                self.__print(0, 1, n._shallow_repr())

        _print_state(self._parameters)
        _print_state(self._buffers)

        # NOTE: The original nn.Module's __call__ method is ignored, which means
        # that hooks of nn.Modules are ignored. It is not recommended
        # to use hooks of nn.Module in nn.Graph for the moment.
        with graph_build_util.DebugScopeContext(
            self.to(GraphModule)._debug_min_s_level,
            self.to(GraphModule)._debug_max_v_level,
            self.to(GraphModule)._debug,
            self.to(GraphModule)._debug_max_py_stack_depth,
            self.to(GraphModule)._debug_only_user_py_stack,
        ):
            result = self.__block_forward(*args, **kwargs)

        outputs = ()
        if not (type(result) is tuple or type(result) is list):
            outputs = (result,)
        else:
            outputs = result

        args_tree = ArgsTree(
            (outputs, {}),
            True,
            "_"
            + self.to(GraphModule).name_prefix
            + self.to(GraphModule).name
            + "_output",
            None,
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
                self.to(GraphModule)._outs_repr.append(out_str)
                self.__print(0, 1, out_str)

        return result

    @property
    def __class__(self):
        if self.to(GraphModule)._belonged_graph._is_user_mode == True:
            return self.to(Module).__class__
        else:
            return type(self)

    def __block_forward(self, *args, **kwargs):
        self.to(GraphModule)._is_executing_forward = True
        args, kwargs = self.__pre_forward_map(*args, **kwargs)
        with self.to(GraphModule).scope_context():
            # "Instance method __func__ is the function object", "when an instance method object is called,
            # the underlying function __func__ is called, inserting the class instance __self__ in front of
            # the argument list."
            # Reference: https://docs.python.org/3/reference/datamodel.html
            unbound_forward_of_module_instance = self.to(Module).forward.__func__
            result = unbound_forward_of_module_instance(self, *args, **kwargs)
        self.to(GraphModule)._is_executing_forward = False
        return result

    def __pre_forward_map(self, *args, **kwargs):
        # Insert identity op when doing activation checkpointing or pipeline execution.
        # Identity op outside activation checkpointing scope will be the endpoint of an activation checkpointing segment.
        # Identity op as the first op of a pipeline stage will make backward op depends on the identity op within the stage,
        # otherwise the backward op may depends the op in former stage which will make graph creates unnessary buffers.
        if self.to(GraphModule)._stage_placement is not None:

            def insert_to_global(t):
                assert isinstance(t, Tensor)
                return self.__get_or_create_global(
                    t, self.to(GraphModule)._stage_placement
                )

            args, kwargs = self.__map_io(
                "input", insert_to_global, "insert_to_global", *args, **kwargs
            )

        if self.to(GraphModule).activation_checkpointing or (
            self.to(GraphModule).stage_id is not None
            and self.to(GraphModule).stage_id >= 0
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
        if key not in self.to(GraphModule)._belonged_graph._unique_global_op_dict:
            # store input tensor to avoid tensor id recycle
            self.to(GraphModule)._belonged_graph._unique_global_op_dict[key] = (
                input_tensor.to_global(placement=placement),
                input_tensor,
            )

        return self.to(GraphModule)._belonged_graph._unique_global_op_dict[key][0]

    def __get_or_create_identity(self, input_tensor: Tensor = None):
        assert input_tensor is not None
        key = input_tensor

        # input_tensor(with placement) -> unique_identity_tensor
        # When placement is different, the input tensor(output tensor of __get_or_create_global) is different, so the
        # key can use only input tensor.
        if key not in self.to(GraphModule)._belonged_graph._unique_identity_op_dict:
            # Reuse current module name for indentity op
            ident_name_scope = graph_build_util.make_new_name_scope(
                self.to(GraphModule).prev_scope,
                self.to(GraphModule).name_prefix + self.to(GraphModule).name,
            )
            with graph_build_util.BlockScopeContext(
                self.to(GraphModule).prev_scope, ident_name_scope
            ):
                # store input tensor to avoid tensor id recycle
                self.to(GraphModule)._belonged_graph._unique_identity_op_dict[
                    key
                ] = oneflow._C.identity(input_tensor)

        return self.to(GraphModule)._belonged_graph._unique_identity_op_dict[key]

    def add_module(self, name: str, module: Optional[Module]) -> None:
        if isinstance(module, Module):
            self.__setattr__(
                name,
                get_block_cls(module)(
                    module,
                    self.to(GraphModule)._name_prefix
                    + self.to(GraphModule)._name
                    + ".",
                    name,
                    self.to(GraphModule)._belonged_graph,
                ),
            )
        elif isinstance(module, Proxy):
            self.__setattr__(name, module)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        self.__setattr__(
            name,
            get_proxy_cls(param)(
                param,
                self.to(GraphModule)._name_prefix + self.to(GraphModule)._name + ".",
                name,
            ),
        )

    def modules(self, memo: Optional[Set["Proxy"]] = None) -> Iterator["Proxy"]:
        assert self.to(GraphModule)._type == GraphBlockType.MODULE
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
            "_"
            + self.to(GraphModule).name_prefix
            + self.to(GraphModule).name
            + "_"
            + io_type,
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

    def __members(self, get_members_fn, recurse=True) -> Iterator["Proxy"]:
        assert self.to(GraphModule)._type == GraphBlockType.MODULE
        memo = set()
        modules = self.modules() if recurse else [self]
        for module in modules:
            members = get_members_fn(module)
            for (k, v) in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                yield v

    def parameters(self, recurse: bool = True) -> Iterator["Proxy"]:
        assert self.to(GraphModule)._type == GraphBlockType.MODULE
        gen = self.__members(lambda module: module._parameters.items(), recurse=recurse)
        for elem in gen:
            yield elem

    def buffers(self, recurse: bool = True) -> Iterator["Proxy"]:
        assert self.to(GraphModule)._type == GraphBlockType.MODULE
        gen = self.__members(lambda module: module._buffers.items(), recurse=recurse)
        for elem in gen:
            yield elem

    def __setattr__(self, name: str, value=None) -> None:
        if value is None or not isinstance(value, Proxy):
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
                            self.to(GraphModule)._name, name
                        )
                    )
            if value.to(GraphModule).type == GraphBlockType.MODULE:
                self._modules[name] = value
            elif value.to(GraphTensor).type == GraphBlockType.PARAMETER:
                self._parameters[name] = value
            elif value.to(GraphTensor).type == GraphBlockType.BUFFER:
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
        if name in self.to(Module)._parameters:
            p_none = self.to(Module)._parameters[name]
            assert p_none is None
            return None
        if name in self.to(Module)._buffers:
            b_none = self.to(Module)._buffers[name]
            assert b_none is None
            return None
        if hasattr(self.to(Module), name):
            # support getting normal attr from the nn.Module
            attr = getattr(self.to(Module), name)
            if isinstance(attr, types.MethodType):
                # If the attr is MethodType, rebind the method to self
                attr = types.MethodType(attr.__func__, self)
            return attr
        raise AttributeError(
            "'{}' '{}' object '{}' in nn.Graph has no attribute '{}'".format(
                self.to(GraphModule)._type,
                type(self).__name__,
                self.to(GraphModule)._name_prefix + self.to(GraphModule).name,
                name,
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
        elif (not graph_build_util.lazy_mode.is_enabled()) and self.to(
            GraphModule
        )._is_executing_forward:
            # eager and inside nn.Graph.build()
            return _s_block.to(Tensor)
        else:
            # outside nn.Graph.build()
            # eager and inside nn.Graph.build()
            return _s_block

    def __repr__(self):
        lines = None
        child_lines = []
        if len(self.to(GraphModule)._args_repr) > 0:
            for in_str in self.to(GraphModule)._args_repr:
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

        if len(self.to(GraphModule)._outs_repr) > 0:
            for out_str in self.to(GraphModule)._outs_repr:
                output_str = add_indent(out_str, 2)
                child_lines.append(output_str)

        child_lines.append(add_indent(repr(self.to(GraphModule)), 2))

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
            + self.to(GraphModule)._type
            + ":"
            + self.to(GraphModule)._name_prefix
            + self.to(GraphModule)._name
            + ":"
            + self._oneflow_internal_origin__._shallow_repr()
            + ")"
        )
        return shallow_repr

    def __print(self, s_level=2, v_level=0, msg: str = ""):
        r"""Do print according to info level.
        """
        assert isinstance(s_level, int)
        assert isinstance(v_level, int)
        assert isinstance(msg, str)
        if s_level >= self.to(GraphModule)._debug_min_s_level:
            if (s_level > 0) or (
                s_level == 0 and v_level <= self.to(GraphModule)._debug_max_v_level
            ):
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
            with block.to(GraphTensor).scope_context():
                self.result = self.method()
            self.finished = True


class ProxyTensor(Proxy):
    def __init__(
        self,
        origin: Union[Parameter, Tensor] = None,
        prefix: str = "",
        name: str = "",
        belonged_graph: weakref.ProxyTypes = None,
    ):
        assert not isinstance(origin, Proxy)
        if isinstance(origin, Parameter):
            self._oneflow_internal_graphblock__ = GraphTensor(
                prefix,
                name,
                belonged_graph,
                weakref.proxy(self),
                GraphBlockType.PARAMETER,
            )
        elif isinstance(origin, Tensor):
            self._oneflow_internal_graphblock__ = GraphTensor(
                prefix, name, belonged_graph, weakref.proxy(self), GraphBlockType.BUFFER
            )
        else:
            raise NotImplementedError()
        self._lazy_origin_builder = LazyBuilder()
        self.build_finished = False
        self._oneflow_internal_graphblock__set_origin(origin)

    def _oneflow_internal_graphblock__set_origin(self, origin):
        self._oneflow_internal_origin__ = origin

    @property
    def lazy_origin(self):
        assert (
            self.to(GraphTensor)._type == GraphBlockType.PARAMETER
            or self.to(GraphTensor)._type == GraphBlockType.BUFFER
        ), "Only Parameter or Buffer Proxy has lazy_origin"
        return self._lazy_origin_builder.result

    def lazy_origin_builder(self):
        assert (
            self.to(GraphTensor)._type == GraphBlockType.PARAMETER
            or self.to(GraphTensor)._type == GraphBlockType.BUFFER
        ), "Only Parameter or Buffer Proxy has lazy_origin_builder"
        return self._lazy_origin_builder

    def set_lazy_origin_builder(self, builder=None):
        assert (
            self.to(GraphTensor)._type == GraphBlockType.PARAMETER
            or self.to(GraphTensor)._type == GraphBlockType.BUFFER
        ), "Only Parameter or Buffer Proxy has lazy_origin_builder"
        self._lazy_origin_builder = builder

    def try_build(self):
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
            + self.to(GraphTensor)._type
            + ":"
            + self.to(GraphTensor)._name_prefix
            + self.to(GraphTensor)._name
            + ":"
            + self._oneflow_internal_origin__._meta_repr()
            + ")"
        )
        return shallow_repr


class ProxySequential(get_seq(ProxyModule)):
    def __init__(
        self,
        origin: Sequential = None,
        prefix: str = "",
        name: str = "",
        belonged_graph: weakref.ProxyTypes = None,
    ):
        super().__init__()
        self.to(GraphModule)._name_prefix = prefix
        self.to(GraphModule)._name = name
        self.to(GraphModule)._belonged_graph = belonged_graph
        self.to(GraphModule)._belonged_block = weakref.proxy(self)
        self._oneflow_internal_graphblock__set_origin(origin)


class ProxyModuleList(get_list(ProxyModule)):
    def __init__(
        self,
        origin: ModuleList = None,
        prefix: str = "",
        name: str = "",
        belonged_graph: weakref.ProxyTypes = None,
    ):
        if isinstance(origin, ModuleList):
            super().__init__()
            self.to(GraphModule)._name_prefix = prefix
            self.to(GraphModule)._name = name
            self.to(GraphModule)._belonged_graph = belonged_graph
            self._oneflow_internal_graphblock__set_origin(origin)
            # ModuleList is a container without forward() method,

        elif isinstance(origin, list):
            super().__init__(origin)
            first = origin[0]
            new_name = "_idx"
            new_list = []
            for item in origin:
                new_name += "-" + item.to(GraphModule).name
                new_list.append(item.to(Module))
            new_module_list = ModuleList(new_list)
            self.to(GraphModule)._name_prefix = (
                first.to(GraphModule).name_prefix + first.to(GraphModule).name
            )
            self.to(GraphModule)._name = new_name
            self.to(GraphModule)._belonged_graph = first.to(GraphModule)._belonged_graph
            self._oneflow_internal_origin__ = new_module_list


class ProxyModuleDict(get_dict(ProxyModule)):
    def __init__(
        self,
        origin: ModuleDict = None,
        prefix: str = "",
        name: str = "",
        belonged_graph: weakref.ProxyTypes = None,
    ):
        super().__init__()
        self.to(GraphModule)._name_prefix = prefix
        self.to(GraphModule)._name = name
        self.to(GraphModule)._belonged_graph = belonged_graph
        self.to(GraphModule)._belonged_block = weakref.proxy(self)
        self._oneflow_internal_graphblock__set_origin(origin)


class ProxyParameterList(get_para_list(ProxyModule)):
    def __init__(
        self,
        origin: ParameterList = None,
        prefix: str = "",
        name: str = "",
        belonged_graph: weakref.ProxyTypes = None,
    ):
        super().__init__()
        self.to(GraphModule)._name_prefix = prefix
        self.to(GraphModule)._name = name
        self.to(GraphModule)._belonged_graph = belonged_graph
        self.to(GraphModule)._belonged_block = weakref.proxy(self)
        self._oneflow_internal_graphblock__set_origin(origin)
        self.to(GraphModule)._is_executing_forward = True

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx = self._get_abs_string_index(idx)
        key = str(idx)
        p_state = self._get_from_states(key, "_parameters")
        if p_state is not None:
            return p_state
        else:
            raise AttributeError("ParameterList dosen't contain ", key)


class ProxyParameterDict(get_para_dict(ProxyModule)):
    def __init__(
        self,
        origin: ParameterDict = None,
        prefix: str = "",
        name: str = "",
        belonged_graph: weakref.ProxyTypes = None,
    ):
        super().__init__()
        self.to(GraphModule)._name_prefix = prefix
        self.to(GraphModule)._name = name
        self.to(GraphModule)._belonged_graph = belonged_graph
        self.to(GraphModule)._belonged_block = weakref.proxy(self)
        self._oneflow_internal_graphblock__set_origin(origin)
        self.to(GraphModule)._is_executing_forward = True

    def __getitem__(self, key: str):
        p_state = self._get_from_states(key, "_parameters")
        if p_state is not None:
            return p_state
        else:
            raise AttributeError("ParameterDict dosen't contain key ", key)
