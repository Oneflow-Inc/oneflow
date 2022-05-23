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
import sys
from collections import OrderedDict
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from oneflow.framework.tensor import Tensor
from string import Template
import google.protobuf as protobuf
from typing import List


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


class IONodeType:
    TENSOR = "TENSOR"
    NONE = "NONE"
    LIST = "LIST"
    TUPLE = "TUPLE"
    DICT = "DICT"
    OPAQUE = "OPAQUE"


class IONode(object):
    def __init__(self, name=None, start_idx=0, value=None, prefix=""):
        # Node indexs
        self._name = name if name is not None else str(start_idx)
        self._prefix = prefix
        self._start_idx = start_idx
        self._end_idx = start_idx
        self._cur_level_idx = -1
        self._sub_nodes = OrderedDict()
        self.attrs = dict()
        self._is_leaf = False

        if isinstance(value, tuple):
            self._type = IONodeType.TUPLE
            self._value = None
            for idx, item in enumerate(value):
                subnode_prefix = (
                    self._prefix
                    + ("." if self._prefix else "")
                    + str(self._cur_level_idx + 1)
                )
                self.__add_sub_node(
                    IONode(None, self._end_idx + 1, item, subnode_prefix)
                )
        elif isinstance(value, list):
            self._type = IONodeType.LIST
            self._value = None
            for idx, item in enumerate(value):
                subnode_prefix = (
                    self._prefix
                    + ("." if self._prefix else "")
                    + str(self._cur_level_idx + 1)
                )
                self.__add_sub_node(
                    IONode(None, self._end_idx + 1, item, subnode_prefix)
                )
        elif isinstance(value, dict):
            self._type = IONodeType.DICT
            self._value = None
            for idx, (key, item) in enumerate(value.items()):
                subnode_prefix = (
                    self._prefix
                    + ("." if self._prefix else "")
                    + str(self._cur_level_idx + 1)
                )
                self.__add_sub_node(
                    IONode(key, self._end_idx + 1, item, subnode_prefix)
                )
        elif isinstance(value, Tensor):
            self._type = IONodeType.TENSOR
            self._value = value
            self._is_leaf = True
        elif value is None:
            self._type = IONodeType.NONE
            self._value = value
            self._is_leaf = True
        else:
            self._type = IONodeType.OPAQUE
            self._value = value
            self._is_leaf = True

    def size(self):
        return self._end_idx - self._start_idx + 1

    def __add_sub_node(self, node):
        self._sub_nodes[self._cur_level_idx + 1] = node
        self._end_idx += node.size()
        self._cur_level_idx += 1

    def named_nodes(self, memo=None):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield (self._prefix + "_" + str(self._name), self)
            for (level_idx, node) in self._sub_nodes.items():
                if node is None:
                    continue
                for n in node.named_nodes(memo):
                    yield n

    def __repr__(self):
        repr_str = ""
        repr_str += "(name: " + self._name
        repr_str += ", idx: " + str(self._start_idx)
        repr_str += ", type: " + self._type
        if self._type == IONodeType.TENSOR:
            repr_str += ", value: " + self._value._meta_repr() + ")"
        else:
            repr_str += ", value: " + repr(self._value) + ")"
        return repr_str

    def map_leaf(self, leaf_node_fn):
        if self._type == IONodeType.TUPLE:
            l_value = list()
            for (name, node) in self._sub_nodes.items():
                l_value.append(node.map_leaf(leaf_node_fn))
            mapped_value = tuple(l_value)
        elif self._type == IONodeType.LIST:
            mapped_value = list()
            for (name, node) in self._sub_nodes.items():
                mapped_value.append(node.map_leaf(leaf_node_fn))
        elif self._type == IONodeType.DICT:
            mapped_value = dict()
            for (name, node) in self._sub_nodes.items():
                mapped_value[node._name] = node.map_leaf(leaf_node_fn)
        else:
            # Leaf node: TENSOR/NONE/OPAQUE
            mapped_value = leaf_node_fn(self)
        return mapped_value
