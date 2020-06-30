# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.util - misc utilities for oneflow.python.onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import shutil
import tempfile

import six
import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_pb2
from oneflow.python.framework import id_util
from google.protobuf import text_format
import onnx
from onnx import helper, onnx_pb, defs, numpy_helper

from . import constants

#
#  mapping dtypes from oneflow to onnx
#
FLOW_TO_ONNX_DTYPE = {
    data_type_pb2.kFloat: onnx_pb.TensorProto.FLOAT,
    data_type_pb2.kDouble: onnx_pb.TensorProto.DOUBLE,
    data_type_pb2.kInt64: onnx_pb.TensorProto.INT64,
    data_type_pb2.kInt32: onnx_pb.TensorProto.INT32,
    data_type_pb2.kInt8: onnx_pb.TensorProto.INT8,
    data_type_pb2.kUInt8: onnx_pb.TensorProto.UINT8,
    data_type_pb2.kFloat16: onnx_pb.TensorProto.FLOAT16,
    # TODO(daquexian): a tempoary hack
    data_type_pb2.kOFRecord: onnx_pb.TensorProto.INT32,
}

#
# mapping dtypes from onnx to numpy
#
ONNX_TO_NUMPY_DTYPE = {
    onnx_pb.TensorProto.FLOAT: np.float32,
    onnx_pb.TensorProto.FLOAT16: np.float16,
    onnx_pb.TensorProto.DOUBLE: np.float64,
    onnx_pb.TensorProto.INT32: np.int32,
    onnx_pb.TensorProto.INT16: np.int16,
    onnx_pb.TensorProto.INT8: np.int8,
    onnx_pb.TensorProto.UINT8: np.uint8,
    onnx_pb.TensorProto.UINT16: np.uint16,
    onnx_pb.TensorProto.INT64: np.int64,
    onnx_pb.TensorProto.UINT64: np.uint64,
    onnx_pb.TensorProto.BOOL: np.bool,
}

#
#  onnx dtype names
#
ONNX_DTYPE_NAMES = {
    onnx_pb.TensorProto.FLOAT: "float",
    onnx_pb.TensorProto.FLOAT16: "float16",
    onnx_pb.TensorProto.DOUBLE: "double",
    onnx_pb.TensorProto.INT32: "int32",
    onnx_pb.TensorProto.INT16: "int16",
    onnx_pb.TensorProto.INT8: "int8",
    onnx_pb.TensorProto.UINT8: "uint8",
    onnx_pb.TensorProto.UINT16: "uint16",
    onnx_pb.TensorProto.INT64: "int64",
    onnx_pb.TensorProto.STRING: "string",
    onnx_pb.TensorProto.BOOL: "bool",
}


def is_integral_onnx_dtype(dtype):
    return dtype in [
        onnx_pb.TensorProto.INT32,
        onnx_pb.TensorProto.INT16,
        onnx_pb.TensorProto.INT8,
        onnx_pb.TensorProto.UINT8,
        onnx_pb.TensorProto.UINT16,
        onnx_pb.TensorProto.INT64,
    ]


class TensorValueInfo(object):
    def __init__(self, tensor_id, g):
        self.id = tensor_id
        if self.id:
            self.dtype = g.get_dtype(tensor_id)
            self.shape = g.get_shape(tensor_id)


ONNX_UNKNOWN_DIMENSION = -1
ONNX_EMPTY_INPUT = ""

def map_flow_dtype(dtype):
    if dtype:
        dtype = FLOW_TO_ONNX_DTYPE[dtype]
    return dtype


def map_numpy_to_onnx_dtype(np_dtype):
    for onnx_dtype, numpy_dtype in ONNX_TO_NUMPY_DTYPE.items():
        if numpy_dtype == np_dtype:
            return onnx_dtype
    raise ValueError("unsupported dtype " + np_dtype + " for mapping")


def map_onnx_to_numpy_type(onnx_type):
    return ONNX_TO_NUMPY_DTYPE[onnx_type]


def make_onnx_shape(shape):
    """shape with -1 is not valid in onnx ... make it a name."""
    if shape:
        # don't do this if input is a scalar
        return [id_util.UniqueStr("unk") if i == -1 else i for i in shape]
    return shape


def make_onnx_inputs_outputs(name, elem_type, shape, **kwargs):
    """Wrapper for creating onnx graph inputs or outputs
       name,  # type: Text
       elem_type,  # type: TensorProto.DataType
       shape,  # type: Optional[Sequence[int]]
    """
    if elem_type is None:
        elem_type = onnx_pb.TensorProto.UNDEFINED
    return helper.make_tensor_value_info(
        name, elem_type, make_onnx_shape(shape), **kwargs
    )


def find_opset(opset):
    """Find opset."""
    if opset is None or opset == 0:
        opset = defs.onnx_opset_version()
        if opset > constants.PREFERRED_OPSET:
            # if we use a newer onnx opset than most runtimes support, default to the one most supported
            opset = constants.PREFERRED_OPSET
    return opset


def get_flow_node_attr(node, name):
    assert node.WhichOneof("op_type") == "user_conf"
    attr_msg = node.user_conf.attr[name]
    attr_type = attr_msg.WhichOneof("value")
    # TODO(daquexian): a better check
    if attr_type == "at_shape":
        return list(getattr(attr_msg, attr_type).dim)
    elif attr_type[:7] == "at_list":
        return list(getattr(attr_msg, attr_type).val)
    else:
        return getattr(attr_msg, attr_type)


def make_sure(bool_val, error_msg, *args):
    if not bool_val:
        raise ValueError("make_sure failure: " + error_msg % args)


def are_shapes_equal(src, dest):
    """ Check whether 2 shapes are equal. """
    if src is None:
        return dest is None
    if dest is None:
        return src is None

    def is_list_or_tuple(obj):
        return isinstance(obj, (list, tuple))
    make_sure(is_list_or_tuple(src), "invalid type for src")
    make_sure(is_list_or_tuple(dest), "invalid type for dest")

    if len(src) != len(dest):
        return False
    return all(i == j for i, j in zip(src, dest))


def get_onnx_version():
    return onnx.__version__


def make_opsetid(domain, version):
    make_sure(isinstance(version, int), "version must be an integer")
    return helper.make_opsetid(domain, version)


def is_onnx_domain(domain):
    if domain is None or domain == "":
        return True
    return False

