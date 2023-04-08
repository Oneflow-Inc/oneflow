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
from typing import Union, List, Tuple, Dict, Callable
from collections import OrderedDict
from oneflow.framework.tensor import Tensor


def _is_raw_type(value, raw_type):
    # Special case for namedtuple return types
    # For example, max(x, dim=1) return oneflow.return_types.max(values=..., indices=...)
    if (
        raw_type == tuple
        and isinstance(value, tuple)
        and type(value).__module__ == "oneflow.return_types"
    ):
        return True
    return type(value) is raw_type


class NamedArg(object):
    r"""
    The class for wrapping over the input/output argument and associating each input/output argument with a prefix and name.
    The input/output argument can be viewed as a tree. NamedArg basically wraps over each tree node on this tree.
    The recursive structure of the input/output arguments are kept, for example:

    input = [1, {key: "value" }] will be constructed into: 
        
    named_input = NamedArg([NamedArg(1), NamedArg({key: NamedArg("value")})])
    """

    def __init__(self, prefix="", name=None, global_index=0) -> None:
        self._name = name if name is not None else str(global_index)
        self._prefix = prefix
        self._global_index = global_index
        self._is_value_set = False
        self._value = None

    def prefix(self):
        return self._prefix

    def name(self):
        return self._name

    def global_index(self):
        return self._global_index

    def value(self):
        assert self._is_value_set, "self._value is not set yet"
        return self._value

    def is_leaf(self):
        assert self._is_value_set, "self._value is not set yet"
        return not (
            _is_raw_type(self._value, dict)
            or _is_raw_type(self._value, OrderedDict)
            or _is_raw_type(self._value, tuple)
            or _is_raw_type(self._value, list)
        )

    def set_value(self, value):
        assert not _is_raw_type(value, NamedArg), "cannot accept value of type NamedArg"
        self._value = value
        self._is_value_set = True

    def __repr__(self):
        repr_str = ""
        repr_str += "(name: " + self._name
        repr_str += ", idx: " + str(self._global_index)
        repr_str += ", type: "
        if _is_raw_type(self._value, tuple):
            repr_str += "TUPLE"
        elif _is_raw_type(self._value, list):
            repr_str += "LIST"
        elif _is_raw_type(self._value, dict) or _is_raw_type(self._value, OrderedDict):
            repr_str += "DICT"
        elif isinstance(self._value, Tensor):
            repr_str += "TENSOR"
        elif self._value is None:
            repr_str += "NONE"
        else:
            repr_str += "OPAQUE"
        if isinstance(self._value, Tensor):
            repr_str += ", value: " + self._value._meta_repr()
        elif (
            _is_raw_type(self._value, dict)
            or _is_raw_type(self._value, OrderedDict)
            or _is_raw_type(self._value, list)
            or _is_raw_type(self._value, tuple)
        ):
            pass
        else:
            repr_str += ", value: " + repr(self._value)
        repr_str += ")"
        return repr_str


class ArgsTree(object):
    def __init__(
        self,
        io_args: Union[Tuple, List, Dict],
        gen_name: bool = False,
        root_prefix: str = "",
        root_name: str = None,
    ) -> None:

        self._io_args = io_args
        self._gen_name = gen_name
        self._root_prefix = root_prefix
        self._root_name = root_name
        self._named_io_args = None
        self._next_global_index = 0

        if self._gen_name:
            self._named_io_args = self._construct_named_io_args(
                self._io_args, self._root_prefix, self._root_name
            )

    def gen_name(self):
        return self._gen_name

    def iter_nodes(self):
        r"""
        return a generator of the args tree nodes in the DFS manner. 
        The node returned can be of type NamedArg or non-NamedArg depending on whether gen_name is set. 
        If gen_name is set, the node will be NamedArg. 
        """

        if self._gen_name:
            args_to_iter = self._named_io_args
        else:
            args_to_iter = self._io_args

        # NOTE(lixiang): Generator expression and iterator are used.
        #   This avoids generating the full list in memory and only processes the nodes that need to be processed,
        #   reducing time and space consumption.
        stack = [iter([args_to_iter])]
        while len(stack) > 0:
            try:
                curr = next(stack[-1])
                if _is_raw_type(curr, NamedArg):
                    curr_value = curr.value()
                else:
                    curr_value = curr

                if _is_raw_type(curr_value, list) or _is_raw_type(curr_value, tuple):
                    children = curr_value
                elif _is_raw_type(curr_value, dict) or _is_raw_type(
                    curr_value, OrderedDict
                ):
                    children = curr_value.values()
                else:
                    children = None

                if children:
                    stack.append(iter(children))

                yield curr

            except StopIteration:
                stack.pop()

    def iter_named_nodes(self):
        assert self._gen_name, "Only use this if gen_name is set!"
        for named_node in self.iter_nodes():
            yield (named_node.prefix() + "_" + named_node.name(), named_node)

    def _construct_named_io_args(self, value, prefix: str, name: str) -> NamedArg:
        arg = NamedArg(prefix, name, self._next_global_index)
        self._next_global_index += 1

        if _is_raw_type(value, list) or _is_raw_type(value, tuple):

            def construct_func(enum):
                (i, v) = enum
                next_prefix = prefix + ("." if prefix else "") + str(i)
                new_arg = self._construct_named_io_args(v, next_prefix, None)
                return new_arg

            arg.set_value(value.__class__(map(construct_func, enumerate(value))))

        elif _is_raw_type(value, dict) or _is_raw_type(value, OrderedDict):

            def construct_func(enum):
                i, (key, v) = enum
                next_prefix = prefix + ("." if prefix else "") + str(i)
                new_arg = self._construct_named_io_args(v, next_prefix, key)
                return key, new_arg

            arg.set_value(
                value.__class__(map(construct_func, enumerate(value.items())))
            )
        else:
            arg.set_value(value)

        return arg

    def map_tuple_leaf(self, map_function: Callable):
        r"""
        When the type of io args is tuple or list, map the leaf of the arguments into map_function(leaf).
        """
        assert map_function != None, "map function cannot be None"
        assert isinstance(
            self._io_args, (tuple, list)
        ), "only used when io args is a tuple or list of tensors"

        stack = []

        # Cases handled: tuple(tensor, ...), such as input args.
        if len(self._io_args) > 0 and isinstance(self._io_args[0], Tensor):
            for i in self._io_args:
                mapped_value = map_function(i)
                stack.append(mapped_value)

            if isinstance(self._io_args, tuple):
                return tuple(stack)
            elif isinstance(self._io_args, list):
                return stack

        # Cases handled: tuple(tuple(tuple(tensor, ...), tuple(tensor, ...), ), etc.
        # Do not loop optimize, and continue to execute the recursive code (`_execute_mapping`).
        elif (
            len(self._io_args) > 0
            and self._io_args[0] is not None
            and not isinstance(self._io_args[0][0], Tensor)
        ):
            return self._execute_mapping(self._io_args, map_function)

        # Cases handled: tuple(tuple(tensor, ...), ), such as the output args of return.
        elif (
            len(self._io_args) > 0
            and isinstance(self._io_args[0], (tuple, list))
            and all(isinstance(arg, Tensor) for arg in self._io_args[0])
        ):
            for i in self._io_args[0]:
                mapped_value = map_function(i)
                stack.append(mapped_value)

            if isinstance(self._io_args[0], tuple):
                return (tuple(stack),)
            elif isinstance(self._io_args[0], list):
                return (stack,)

        # Other cases.
        # Do not loop optimize, and continue to execute the recursive code (`_execute_mapping`).
        else:
            return self._execute_mapping(self._io_args, map_function)

    def map_leaf(self, map_function: Callable):
        r"""
        Map the leaf of the arguments into map_function(leaf).
        """
        assert map_function != None, "map function cannot be None"

        if self._gen_name:
            args_to_map = self._named_io_args
        else:
            args_to_map = self._io_args

        return self._execute_mapping(args_to_map, map_function)

    def _execute_mapping(self, value, map_function):
        if _is_raw_type(value, tuple) or _is_raw_type(value, list):
            mapped_value = value.__class__(
                map(lambda x: self._execute_mapping(x, map_function), value)
            )
        elif _is_raw_type(value, dict) or _is_raw_type(value, OrderedDict):
            mapped_value = value.__class__(
                map(
                    lambda x: (x[0], self._execute_mapping(x[1], map_function)),
                    value.items(),
                )
            )
        elif _is_raw_type(value, NamedArg):
            if value.is_leaf():  # only map the leaf: TENSOR/NONE/OPAQUE
                mapped_value = map_function(value)
            else:
                mapped_value = self._execute_mapping(value.value(), map_function)
        else:
            mapped_value = map_function(value)

        return mapped_value
