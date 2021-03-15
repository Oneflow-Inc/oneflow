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
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# oneflow.python.onnx.util - misc utilities for oneflow.python.onnx

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import re
import shutil
import tempfile

from google.protobuf import text_format
import numpy as np
import onnx
from onnx import helper, onnx_pb, defs, numpy_helper
import six

from oneflow.python.framework import id_util
from oneflow.python.onnx import constants
import oneflow
import oneflow_api


#
#  mapping dtypes from oneflow to onnx
#
FLOW_2_ONNX_DTYPE = {
    oneflow.float32: onnx_pb.TensorProto.FLOAT,
    oneflow.float64: onnx_pb.TensorProto.DOUBLE,
    oneflow.int64: onnx_pb.TensorProto.INT64,
    oneflow.int32: onnx_pb.TensorProto.INT32,
    oneflow.int8: onnx_pb.TensorProto.INT8,
    oneflow.uint8: onnx_pb.TensorProto.UINT8,
    oneflow.float16: onnx_pb.TensorProto.FLOAT16,
}

FLOW_PROTO_2_ONNX_DTYPE = {}
for k, v in FLOW_2_ONNX_DTYPE.items():
    FLOW_PROTO_2_ONNX_DTYPE[oneflow_api.deprecated.GetProtoDtype4OfDtype(k)] = v
del k

#
# mapping dtypes from onnx to numpy
#
ONNX_2_NUMPY_DTYPE = {
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


ONNX_UNKNOWN_DIMENSION = -1
ONNX_EMPTY_INPUT = ""


def Flow2OnnxDtype(dtype):
    assert dtype in FLOW_2_ONNX_DTYPE or dtype in FLOW_PROTO_2_ONNX_DTYPE
    if dtype in FLOW_2_ONNX_DTYPE:
        return FLOW_2_ONNX_DTYPE[dtype]
    else:
        return FLOW_PROTO_2_ONNX_DTYPE[dtype]


def Onnx2FlowDtype(dtype):
    for flow_dtype, onnx_dtype in FLOW_2_ONNX_DTYPE.items():
        if onnx_dtype == dtype:
            return flow_dtype
    raise ValueError("unsupported dtype " + np_dtype + " for mapping")


def Numpy2OnnxDtype(np_dtype):
    for onnx_dtype, numpy_dtype in ONNX_2_NUMPY_DTYPE.items():
        if numpy_dtype == np_dtype:
            return onnx_dtype
    raise ValueError("unsupported dtype " + np_dtype + " for mapping")


def Onnx2NumpyDtype(onnx_type):
    return ONNX_2_NUMPY_DTYPE[onnx_type]


def MakeOnnxShape(shape):
    """shape with -1 is not valid in onnx ... make it a name."""
    if shape:
        # don't do this if input is a scalar
        return [id_util.UniqueStr("unk") if i == -1 else i for i in shape]
    return shape


def MakeOnnxInputsOutputs(name, elem_type, shape, **kwargs):
    """Wrapper for creating onnx graph inputs or outputs
       name,  # type: Text
       elem_type,  # type: TensorProto.DataType
       shape,  # type: Optional[Sequence[int]]
    """
    if elem_type is None:
        elem_type = onnx_pb.TensorProto.UNDEFINED
    return helper.make_tensor_value_info(
        name, elem_type, MakeOnnxShape(shape), **kwargs
    )


def FindOpset(opset):
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


def MakeSure(bool_val, error_msg, *args):
    if not bool_val:
        raise ValueError("MakeSure failure: " + error_msg % args)


def AreShapesEqual(src, dest):
    """ Check whether 2 shapes are equal. """
    if src is None:
        return dest is None
    if dest is None:
        return src is None

    def is_list_or_tuple(obj):
        return isinstance(obj, (list, tuple))

    MakeSure(is_list_or_tuple(src), "invalid type for src")
    MakeSure(is_list_or_tuple(dest), "invalid type for dest")

    if len(src) != len(dest):
        return False
    return all(i == j for i, j in zip(src, dest))


def get_onnx_version():
    return onnx.__version__


def is_onnx_domain(domain):
    if domain is None or domain == "":
        return True
    return False


def GenerateValidFilename(s):
    return "".join([c if c.isalpha() or c.isdigit() else "_" for c in s])


def TensorProtoFromNumpy(
    arr: np.ndarray, name=None, external_data=False, export_path=None
):
    if name is None:
        name = id_util.UniqueStr("tensor_")
    tp = numpy_helper.from_array(arr, name)
    # value with size < 1024 bytes will remain in .onnx file
    # (like what pytorch does)
    if (not external_data) or arr.nbytes < 1024:
        return tp
    assert tp.HasField("raw_data")
    tp.ClearField("raw_data")
    export_dir = os.path.dirname(export_path)
    filename = GenerateValidFilename(name)
    with open(os.path.join(export_dir, filename), "wb") as f:
        arr.tofile(f)
    tp.data_location = onnx_pb.TensorProto.EXTERNAL
    external_data = tp.external_data.add()
    external_data.key = "location"
    external_data.value = filename
    return tp
