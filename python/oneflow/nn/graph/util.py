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
            sub_arg_repr_list.append(lbns[bn_idx] + ":(" + sub_bns_sbp[bn_idx] + ", " + sub_bns_desc[bn_idx] + ")")


        if len(lbns) > 1:  # arg of multiple tensors
            arg_repr_list.append("[" + (", ").join(sub_arg_repr_list) + "]")
        else:
            assert len(lbns) == 1
            arg_repr_list.append(sub_arg_repr_list[0])

    return arg_repr_list

def _get_user_op_io_repr(op_conf, bn2nd_sbp, lbn2blob_desc):
    user_op_conf = op_conf.user_conf
    input_sig_str = ", ".join(_get_args_repr(user_op_conf.input_order, user_op_conf.input, bn2nd_sbp, lbn2blob_desc))
    output_sig_str = ", ".join(_get_args_repr(user_op_conf.output_order, user_op_conf.output, bn2nd_sbp, lbn2blob_desc))
    return input_sig_str, output_sig_str

def _get_var_op_io_repr(op_conf, bn2nd_sbp, lbn2blob_desc):
    input_sig_str = ""
    var_op_conf = op_conf.variable_conf
    output_lbn = op_conf.name + "/" + var_op_conf.out
    output_sig_str = var_op_conf.out
    nd_sbp = bn2nd_sbp[var_op_conf.out]
    output_sig_str += ":" + _nd_sbp2repr(nd_sbp)  + ", " + _blob_desc_repr(lbn2blob_desc[output_lbn])
    return input_sig_str, output_sig_str

def _get_iden_op_io_repr(op_conf, bn2nd_sbp, lbn2blob_desc):
    iden_op_conf = op_conf.identity_conf
    input_lbn = getattr(iden_op_conf, "in")
    input_sig_str = input_lbn + ":" + _nd_sbp2repr(bn2nd_sbp["in"])  + ", " + _blob_desc_repr(lbn2blob_desc[input_lbn])

    output_lbn = op_conf.name + "/" + iden_op_conf.out
    output_sig_str = iden_op_conf.out
    nd_sbp = bn2nd_sbp[iden_op_conf.out]
    output_sig_str += ":" + _nd_sbp2repr(nd_sbp)  + ", " + _blob_desc_repr(lbn2blob_desc[output_lbn])

    return input_sig_str, output_sig_str


def operators_repr(ops, graph_proto):
    r"""Generate operators' string representation of this module
    """
    if len(ops) > 0:
        op_confs = dict()
        for op_conf in graph_proto.net.op:
            op_confs[op_conf.name] = op_conf

    def _op_signature(op):
        bn2nd_sbp = graph_proto.job_parallel_view_conf.op_name2nd_sbp_signature_conf[op.name].bn_in_op2nd_sbp
        lbn2blob_desc = graph_proto.helper.lbn2logical_blob_desc
        signature_template = Template(op.name + "($input) -> ($output)")
        input_sig_str = "..."
        output_sig_str = "..."

        # Only deal with UserOpConf and VariableOpConf for now.
        if op.HasField("user_conf"):
            input_sig_str, output_sig_str = _get_user_op_io_repr(op, bn2nd_sbp, lbn2blob_desc)
        elif op.HasField("variable_conf"):
            input_sig_str, output_sig_str = _get_var_op_io_repr(op, bn2nd_sbp, lbn2blob_desc)
        elif op.HasField("identity_conf"):
            input_sig_str, output_sig_str = _get_iden_op_io_repr(op, bn2nd_sbp, lbn2blob_desc)
        elif op.name.startswith("System-"):
            return False, ""

        op_str = "(OPERATOR: "
        op_str += signature_template.substitute(input=input_sig_str, output=output_sig_str)
        op_str += ")"

        return True, op_str

    ops_strs = []
    for op in ops:
        assert op in op_confs
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
