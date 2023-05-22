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
from typing import Callable, Dict, Union, List, Tuple, Optional

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


class GraphIR(object):
    def __init__(self, g_proto: job_pb.Job):
        assert g_proto is not None and isinstance(g_proto, job_pb.Job)
        self._graph_proto = g_proto
        self._op2conf = None
        self._op2placement = None

    def get_op_conf(self, op_name: str) -> Optional[op_conf_util.OperatorConf]:
        if self._op2conf is None:
            self._op2conf = dict()
            for op_conf in self._graph_proto.net.op:
                self._op2conf[op_conf.name] = op_conf
        if op_name not in self._op2conf:
            return None
        return self._op2conf[op_name]

    def get_op_placement(self, op_name: str) -> Optional[oneflow.placement]:
        if self._op2placement is None:
            self._op2placement = dict()
            for group in self._graph_proto.placement.placement_group:
                parallel_conf = group.parallel_conf
                for this_op_name in group.op_set.op_name:
                    self._op2placement[this_op_name] = oneflow.placement(
                        proto_str=text_format.MessageToString(parallel_conf)
                    )
        if op_name not in self._op2placement:
            return None
        return self._op2placement[op_name]


def _op_signature(
    op: op_conf_util.OperatorConf,
    graph_proto: job_pb.Job,
    graph_ir: GraphIR,
    show_op_loc: bool,
) -> Tuple[bool, str]:
    bn2nd_sbp = graph_proto.job_parallel_view_conf.op_name2nd_sbp_signature_conf[
        op.name
    ].bn_in_op2nd_sbp
    lbn2blob_desc = graph_proto.helper.lbn2logical_blob_desc
    signature_template = Template(
        op.name
        + "($input) -> ($output)"
        + ", placement=("
        + str(graph_ir.get_op_placement(op.name))
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
    op_str += signature_template.substitute(input=input_sig_str, output=output_sig_str)

    if show_op_loc and op.loc:
        op_str += ", location=(" + op.loc + ")"

    op_str += ")"

    return True, op_str


def operators_repr(
    ops: protobuf.pyext._message.RepeatedCompositeContainer,
    graph_ir: GraphIR,
    show_op_loc: bool,
) -> List[str]:
    r"""Generate operators' string representation of this module
    """
    graph_proto = graph_ir._graph_proto
    ops_strs = []
    for op in ops:
        op_conf = graph_ir.get_op_conf(op)
        if op_conf is None:
            continue
        assert isinstance(op_conf, op_conf_util.OperatorConf)
        got_repr, op_str = _op_signature(op_conf, graph_proto, graph_ir, show_op_loc)
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
