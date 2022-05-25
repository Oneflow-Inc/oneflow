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
from os import abort
import sys
from collections import OrderedDict
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from oneflow.framework.tensor import Tensor
from typing import Callable, Dict, Union, List, Tuple
from string import Template
import google.protobuf as protobuf


def operators_repr(
    ops: protobuf.pyext._message.RepeatedCompositeContainer,
) -> List[str]:
    r"""Generate operators' string representation
    """

    def _op_signature(op: op_conf_util.OperatorConf) -> str:

        signature_template = Template(op.name + "($input) -> ($output)")
        input_sig_str = "..."
        output_sig_str = "..."

        # only deal with UserOpConf and VariableOpConf for now
        if op.HasField("user_conf"):
            user_conf = op.user_conf
            input_params = []
            for param in user_conf.input_order:
                x = user_conf.input[param].s
                if len(x) > 1:  # param of multiple tensors
                    input_params.append("[" + (", ").join(list(x)) + "]")
                else:
                    assert len(x) == 1
                    input_params.append(x[0])
            input_sig_str = ", ".join(input_params)

            output_params = []
            for param in user_conf.output_order:
                x = user_conf.output[param].s
                if len(x) > 1:
                    output_params.append("[" + (", ").join(list(x)) + "]")
                else:
                    assert len(x) == 1
                    output_params.append(x[0])
            output_sig_str = ", ".join(output_params)

        elif op.HasField("variable_conf"):
            variable_conf = op.variable_conf
            input_sig_str = ""
            output_sig_str = op.name + "/" + variable_conf.out

        return signature_template.substitute(input=input_sig_str, output=output_sig_str)

    return map(lambda op: "(OPERATOR: " + _op_signature(op) + ")", ops)


def add_indent(in_s, num_spaces):
    s = in_s.split("\n")
    if len(s) == 1:
        return in_s
    first = s.pop(0)
    s = [num_spaces * " " + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def sys_exc_error_msg():
    msg = ""
    exc_info = sys.exc_info()
    if len(exc_info) > 0:
        msg += str(exc_info[0])
    if len(exc_info) > 1:
        msg += " " + str(exc_info[1])
    return msg


def seq_to_func_return(seq, need_unpack=False):
    if need_unpack:
        return seq[0]
    return seq

class IOArgs(object):
    def __init__(self, io_args: Union[Tuple, List, Dict], gen_name: bool = False, root_prefix:str = "", root_name:str = None) -> None:
        assert (
            isinstance(io_args, dict)
            or isinstance(io_args, tuple)
            or isinstance(io_args, list)
        ), "input/output arguments must be one of those types"

        self._io_args = io_args 
        self._gen_name = gen_name
        self._root_prefix = root_prefix
        self._root_name = root_name 
        self._named_io_args = None 
        self._flattened_named_args = None 

        if self._gen_name:
            self._named_io_args = self._construct_named_io_args()

    def gen_name(self):
        return self._gen_name

    class NamedIOArg(object):
        r"""
        The class for wrapping over the input/output argument and associating each input/output argument with a prefix and name.
        The input/output argument can be viewed as a tree. NamedIOArg basically wraps over each tree node on this tree.
        The recursive structure of the input/output arguments are kept, for example:

        iuput = [1, {key: "value" }] will be constructed into: 
            input_args = NamedIOArg([NamedIOArg(1), NamedIOArg({key: NamedIOArg("value")})])
        """
        def __init__(self, prefix = "", name = None, global_index = 0) -> None:
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
                isinstance(self._value, dict)
                or isinstance(self._value, tuple)
                or isinstance(self._value, list)
            )

        def set_value(self, value):
            assert not isinstance(
                value, IOArgs.NamedIOArg
            ), "cannot accept value of type NamedIOArg"
            self._value = value
            self._is_value_set = True

        def __repr__(self):
            repr_str = ""
            repr_str += "(name: " + self._name
            repr_str += ", idx: " + str(self._global_index)
            repr_str += ", type: "
            if isinstance(self._value, tuple):
                repr_str += "TUPLE"
            elif isinstance(self._value, list):
                repr_str += "LIST"
            elif isinstance(self._value, dict):
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
                isinstance(self._value, dict)
                or isinstance(self._value, list)
                or isinstance(self._value, tuple)
            ):
                pass
            else:
                repr_str += ", value: " + repr(self._value)
            repr_str += ")"
            return repr_str
        
    def _construct_named_io_args(self):
        global_index = 0
        flattended_named_args = []

        def construct(value, prefix: str, name: str) -> IOArgs.NamedIOArg:
            nonlocal global_index
            nonlocal flattended_named_args
    
            arg = IOArgs.NamedIOArg(prefix, name, global_index)
            flattended_named_args.append((arg.prefix() + "_" + arg.name(), arg))
            global_index += 1

            if isinstance(value, list) or isinstance(value, tuple):

                def construct_func(enum):
                    (i, v) = enum
                    next_prefix = prefix + ("." if prefix else "") + str(i)
                    new_arg = construct(v, next_prefix, None)
                    return new_arg

                arg.set_value(value.__class__(map(construct_func, enumerate(value))))

            elif isinstance(value, dict):

                def construct_func(enum):
                    i, (key, v) = enum
                    next_prefix = prefix + ("." if prefix else "") + str(i)
                    new_arg = construct(v, next_prefix, key)
                    return key, new_arg

                arg.set_value(
                    value.__class__(map(construct_func, enumerate(value.items())))
                )
            else:
                arg.set_value(value)
            return arg

        root_arg = construct(self._io_args, self._root_prefix, self._root_name)
        self._flattened_named_args = flattended_named_args

        return root_arg

    def flattened_named_args(self):
        assert self._gen_name, "only call this method if gen_name is set"
        return self._flattened_named_args

    def map_leaf(self, map_function: Callable):
        r"""
        Map the leaf of the IO arguments into map_function(leaf).
        """
        assert map_function != None, "map function cannot be None"
        
        if self._gen_name:
            args_to_map = self._named_io_args
        else:
            args_to_map = self._io_args

        def execute_mapping(value):
            if isinstance(value, tuple) or isinstance(value, list):
                mapped_value = value.__class__(map(lambda x: execute_mapping(x), value))
            elif isinstance(value, dict):
                mapped_value = value.__class__(
                    map(lambda x: (x[0], execute_mapping(x[1])), value.items())
                )
            elif isinstance(value, IOArgs.NamedIOArg):
                if value.is_leaf():  # only map the leaf: TENSOR/NONE/OPAQUE
                    mapped_value = map_function(value)
                else:
                    mapped_value = execute_mapping(value.value())
            else:
                mapped_value = map_function(value)

            return mapped_value

        return execute_mapping(args_to_map)
        