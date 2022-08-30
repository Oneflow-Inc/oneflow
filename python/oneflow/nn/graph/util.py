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
from string import Template
from collections import OrderedDict
from typing import Callable, Dict, Union, List, Tuple

import google.protobuf as protobuf
from google.protobuf import text_format

import oneflow
import oneflow.core.job.job_pb2 as job_pb
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from oneflow.framework.tensor import Tensor


def _nd_sbp2repr(nd_sbp):
    dim_len = len(nd_sbp.sbp_parallel)
    nd_sbp_str = "sbp=("
    for i in range(dim_len):
        if i > 0:
            nd_sbp_str += ", "
        sbp = nd_sbp.sbp_parallel[i]
        if sbp.HasField("broadcast_parallel"):
            nd_sbp_str += "B"
        elif sbp.HasField("partial_sum_parallel"):
            nd_sbp_str += "P"
        elif sbp.HasField("split_parallel"):
            nd_sbp_str += "S(" + str(sbp.split_parallel.axis) + ")"
    nd_sbp_str += ")"
    return nd_sbp_str


def _blob_desc_repr(blob_desc):
    desc_str = "size=("
    for i in range(len(blob_desc.shape.dim)):
        if i > 0:
            desc_str += ", "
        desc_str += str(blob_desc.shape.dim[i])
    desc_str += "), "
    desc_str += "dtype=("
    desc_str += str(oneflow.dtype.get(int(blob_desc.data_type)))
    desc_str += ")"
    return desc_str


def _get_args_repr(ordered_bn, bn2lbn, bn2nd_sbp, lbn2blob_desc):
    arg_repr_list = []
    for bn in ordered_bn:
        lbns = list(bn2lbn[bn].s)

        # sbp repr
        sub_bns_sbp = []
        for bn_idx in range(len(lbns)):
            sub_bn = bn + "_" + str(bn_idx)
            nd_sbp = bn2nd_sbp[sub_bn]
            sub_bns_sbp.append(_nd_sbp2repr(nd_sbp))

        # TODO: placement repr

        # shape repr and dtype
        sub_bns_desc = []
        for bn_idx in range(len(lbns)):
            sub_bns_desc.append(_blob_desc_repr(lbn2blob_desc[lbns[bn_idx]]))

        # sub arg repr
        sub_arg_repr_list = []
        for bn_idx in range(len(lbns)):
            sub_arg_repr_list.append(
                lbns[bn_idx]
                + ":("
                + sub_bns_sbp[bn_idx]
                + ", "
                + sub_bns_desc[bn_idx]
                + ")"
            )

        if len(lbns) > 1:  # arg of multiple tensors
            arg_repr_list.append("[" + (", ").join(sub_arg_repr_list) + "]")
        else:
            assert len(lbns) == 1
            arg_repr_list.append(sub_arg_repr_list[0])

    return arg_repr_list


def _get_user_op_io_repr(op_conf, bn2nd_sbp, lbn2blob_desc):
    user_op_conf = op_conf.user_conf
    input_sig_str = ", ".join(
        _get_args_repr(
            user_op_conf.input_order, user_op_conf.input, bn2nd_sbp, lbn2blob_desc
        )
    )
    output_sig_str = ", ".join(
        _get_args_repr(
            user_op_conf.output_order, user_op_conf.output, bn2nd_sbp, lbn2blob_desc
        )
    )
    return input_sig_str, output_sig_str


def _get_var_op_io_repr(op_conf, bn2nd_sbp, lbn2blob_desc):
    input_sig_str = ""
    var_op_conf = op_conf.variable_conf
    output_lbn = op_conf.name + "/" + var_op_conf.out
    output_sig_str = var_op_conf.out
    nd_sbp = bn2nd_sbp[var_op_conf.out]
    output_sig_str += (
        ":" + _nd_sbp2repr(nd_sbp) + ", " + _blob_desc_repr(lbn2blob_desc[output_lbn])
    )
    return input_sig_str, output_sig_str


def _get_iden_op_io_repr(op_conf, bn2nd_sbp, lbn2blob_desc):
    iden_op_conf = op_conf.identity_conf
    input_lbn = getattr(iden_op_conf, "in")
    input_sig_str = (
        input_lbn
        + ":"
        + _nd_sbp2repr(bn2nd_sbp["in"])
        + ", "
        + _blob_desc_repr(lbn2blob_desc[input_lbn])
    )

    output_lbn = op_conf.name + "/" + iden_op_conf.out
    output_sig_str = iden_op_conf.out
    nd_sbp = bn2nd_sbp[iden_op_conf.out]
    output_sig_str += (
        ":" + _nd_sbp2repr(nd_sbp) + ", " + _blob_desc_repr(lbn2blob_desc[output_lbn])
    )

    return input_sig_str, output_sig_str


def _get_input_op_io_repr(op_conf, bn2nd_sbp, lbn2blob_desc):
    op_input_conf = op_conf.input_conf
    output_lbn = op_conf.name + "/" + op_input_conf.out
    nd_sbp = bn2nd_sbp[op_input_conf.out]
    output_sig_str = (
        output_lbn
        + ":"
        + _nd_sbp2repr(nd_sbp)
        + ", "
        + _blob_desc_repr(lbn2blob_desc[output_lbn])
    )
    return "", output_sig_str


def _get_output_op_io_repr(op_conf, bn2nd_sbp, lbn2blob_desc):
    op_output_conf = op_conf.output_conf
    input_lbn = getattr(op_output_conf, "in")
    output_lbn = op_conf.name + "/" + op_output_conf.out

    input_sig_str = (
        input_lbn
        + ":"
        + _nd_sbp2repr(bn2nd_sbp["in"])
        + ", "
        + _blob_desc_repr(lbn2blob_desc[output_lbn])
    )

    nd_sbp = bn2nd_sbp[op_output_conf.out]
    output_sig_str = (
        output_lbn
        + ":"
        + _nd_sbp2repr(nd_sbp)
        + ", "
        + _blob_desc_repr(lbn2blob_desc[output_lbn])
    )
    return input_sig_str, output_sig_str


def operators_repr(
    ops: protobuf.pyext._message.RepeatedCompositeContainer,
    graph_proto: job_pb.Job,
    show_op_loc: bool,
) -> List[str]:
    r"""Generate operators' string representation of this module
    """
    if len(ops) > 0:
        op_confs = dict()
        for op_conf in graph_proto.net.op:
            op_confs[op_conf.name] = op_conf

        op2placement = dict()
        for group in graph_proto.placement.placement_group:
            parallel_conf = group.parallel_conf
            for op_name in group.op_set.op_name:
                op2placement[op_name] = str(
                    oneflow.placement(
                        proto_str=text_format.MessageToString(parallel_conf)
                    )
                )

    def _op_signature(op: op_conf_util.OperatorConf) -> Tuple[bool, str]:
        bn2nd_sbp = graph_proto.job_parallel_view_conf.op_name2nd_sbp_signature_conf[
            op.name
        ].bn_in_op2nd_sbp
        lbn2blob_desc = graph_proto.helper.lbn2logical_blob_desc
        signature_template = Template(
            op.name
            + "($input) -> ($output)"
            + ", placement=("
            + op2placement[op.name]
            + ")"
        )
        input_sig_str = "..."
        output_sig_str = "..."

        # Only deal with UserOpConf and VariableOpConf for now.
        if op.HasField("user_conf"):
            input_sig_str, output_sig_str = _get_user_op_io_repr(
                op, bn2nd_sbp, lbn2blob_desc
            )
        elif op.HasField("variable_conf"):
            input_sig_str, output_sig_str = _get_var_op_io_repr(
                op, bn2nd_sbp, lbn2blob_desc
            )
        elif op.HasField("identity_conf"):
            input_sig_str, output_sig_str = _get_iden_op_io_repr(
                op, bn2nd_sbp, lbn2blob_desc
            )
        elif op.HasField("input_conf"):
            input_sig_str, output_sig_str = _get_input_op_io_repr(
                op, bn2nd_sbp, lbn2blob_desc
            )
        elif op.HasField("output_conf"):
            input_sig_str, output_sig_str = _get_output_op_io_repr(
                op, bn2nd_sbp, lbn2blob_desc
            )
        elif op.name.startswith("System-"):
            return False, ""

        op_str = "(OPERATOR: "
        op_str += signature_template.substitute(
            input=input_sig_str, output=output_sig_str
        )

        if show_op_loc and op.loc:
            op_str += ", location=(" + op.loc + ")"

        op_str += ")"

        return True, op_str

    ops_strs = []
    for op in ops:
        if op not in op_confs:
            continue
        op_conf = op_confs[op]
        assert isinstance(op_conf, op_conf_util.OperatorConf)
        got_repr, op_str = _op_signature(op_conf)
        if got_repr:
            ops_strs.append(op_str)
    return ops_strs


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


class NamedArg(object):
    r"""
    The class for wrapping over the input/output argument and associating each input/output argument with a prefix and name.
    The input/output argument can be viewed as a tree. NamedArg basically wraps over each tree node on this tree.
    The recursive structure of the input/output arguments are kept, for example:

    iuput = [1, {key: "value" }] will be constructed into: 
        
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
            isinstance(self._value, dict)
            or isinstance(self._value, tuple)
            or isinstance(self._value, list)
        )

    def set_value(self, value):
        assert not isinstance(value, NamedArg), "cannot accept value of type NamedArg"
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


class ArgsTree(object):
    def __init__(
        self,
        io_args: Union[Tuple, List, Dict],
        gen_name: bool = False,
        root_prefix: str = "",
        root_name: str = None,
    ) -> None:
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

        stack = []
        stack.append(args_to_iter)
        while len(stack) > 0:
            curr = stack.pop()
            if isinstance(curr, NamedArg):
                curr_value = curr.value()
            else:
                curr_value = curr

            if isinstance(curr_value, list) or isinstance(curr_value, tuple):
                children = curr_value
            elif isinstance(curr_value, dict):
                children = list(curr_value.values())
            else:
                children = None

            if children:
                for child in reversed(children):
                    stack.append(child)

            yield curr

    def iter_named_nodes(self):
        assert self._gen_name, "Only use this if gen_name is set!"
        for named_node in self.iter_nodes():
            yield (named_node.prefix() + "_" + named_node.name(), named_node)

    def _construct_named_io_args(self, value, prefix: str, name: str) -> NamedArg:
        arg = NamedArg(prefix, name, self._next_global_index)
        self._next_global_index += 1

        if isinstance(value, list) or isinstance(value, tuple):

            def construct_func(enum):
                (i, v) = enum
                next_prefix = prefix + ("." if prefix else "") + str(i)
                new_arg = self._construct_named_io_args(v, next_prefix, None)
                return new_arg

            arg.set_value(value.__class__(map(construct_func, enumerate(value))))

        elif isinstance(value, dict):

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
        if isinstance(value, tuple) or isinstance(value, list):
            mapped_value = value.__class__(
                map(lambda x: self._execute_mapping(x, map_function), value)
            )
        elif isinstance(value, dict):
            mapped_value = value.__class__(
                map(
                    lambda x: (x[0], self._execute_mapping(x[1], map_function)),
                    value.items(),
                )
            )
        elif isinstance(value, NamedArg):
            if value.is_leaf():  # only map the leaf: TENSOR/NONE/OPAQUE
                mapped_value = map_function(value)
            else:
                mapped_value = self._execute_mapping(value.value(), map_function)
        else:
            mapped_value = map_function(value)

        return mapped_value
