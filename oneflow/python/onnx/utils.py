# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.utils - misc utilities for oneflow.python.onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import shutil
import tempfile
from distutils.version import LooseVersion

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import six
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import types_pb2, tensor_pb2
from tensorflow.python.framework import tensor_util
import oneflow.core.common.data_type_pb2 as data_type_pb2
from google.protobuf import text_format
import onnx
from onnx import helper, onnx_pb, defs, numpy_helper

from . import constants

#
#  mapping dtypes from tensorflow to onnx
#
TF_TO_ONNX_DTYPE = {
    data_type_pb2.kFloat: onnx_pb.TensorProto.FLOAT,
    data_type_pb2.kDouble: onnx_pb.TensorProto.DOUBLE,
    data_type_pb2.kInt64: onnx_pb.TensorProto.INT64,
    data_type_pb2.kInt32: onnx_pb.TensorProto.INT32,
    data_type_pb2.kInt8: onnx_pb.TensorProto.INT8,
    data_type_pb2.kUInt8: onnx_pb.TensorProto.UINT8,
    data_type_pb2.kFloat16: onnx_pb.TensorProto.FLOAT16,
    #TODO(daquexian): a tempoary hack
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
    onnx_pb.TensorProto.BOOL: "bool"
}


class TensorValueInfo(object):
    def __init__(self, tensor_id, g):
        self.id = tensor_id
        if self.id:
            self.dtype = g.get_dtype(tensor_id)
            self.shape = g.get_shape(tensor_id)


ONNX_UNKNOWN_DIMENSION = -1
ONNX_EMPTY_INPUT = ""

# index for internally generated names
INTERNAL_NAME = 1

# Fake onnx op type which is used for Graph input.
GRAPH_INPUT_TYPE = "NON_EXISTENT_ONNX_TYPE"


def make_name(name):
    """Make op name for inserted ops."""
    global INTERNAL_NAME
    INTERNAL_NAME += 1
    return "{}__{}".format(name, INTERNAL_NAME)


def split_nodename_and_shape(name):
    """input name with shape into name and shape."""
    # pattern for a node name
    inputs = []
    shapes = {}
    # input takes in most cases the format name:0, where 0 is the output number
    # in some cases placeholders don't have a rank which onnx can't handle so we let uses override the shape
    # by appending the same, ie : [1,28,28,3]
    name_pattern = r"(?:([\w\d/\-\._:]+)(\[[\-\d,]+\])?),?"
    splits = re.split(name_pattern, name)
    for i in range(1, len(splits), 3):
        inputs.append(splits[i])
        if splits[i + 1] is not None:
            shapes[splits[i]] = [int(n) for n in splits[i + 1][1:-1].split(",")]
    if not shapes:
        shapes = None
    return inputs, shapes


def tf_to_onnx_tensor(tensor, name=""):
    """Convert tensorflow tensor to onnx tensor."""
    np_data = get_tf_tensor_data(tensor)
    if np_data.dtype == np.object:
        # assume np_data is string, numpy_helper.from_array accepts ndarray,
        # in which each item is of str while the whole dtype is of object.
        try:
            np_data = np_data.astype(np.str).astype(np.object)
        except: # pylint: disable=bare-except
            raise RuntimeError("Not support type: {}".format(type(np_data.flat[0])))
    return numpy_helper.from_array(np_data, name=name)


def get_tf_tensor_data(tensor):
    """Get data from tensor."""
    make_sure(isinstance(tensor, tensor_pb2.TensorProto), "Require TensorProto")
    np_data = tensor_util.MakeNdarray(tensor)
    make_sure(isinstance(np_data, np.ndarray), "{} isn't ndarray".format(np_data))
    return np_data


def get_tf_const_value(op, as_list=True):
    """
    If as_list=True, return the array as a (possibly nested) list.
    Otherwise, return data of type np.ndarray.

    If a tensor is a scalar having value 1,
        when as_list=False, return np.array(1), type is <class 'numpy.ndarray'>
        when as_list=True, return 1, type is <class 'int'>.
    """
    make_sure(is_tf_const_op(op), "{} isn't a const op".format(op.name))
    value = get_tf_tensor_data(op.get_attr("value"))
    if as_list:
        value = value.tolist()
    return value


def get_tf_shape_attr(node):
    """Get shape from tensorflow attr "shape"."""
    dims = None
    try:
        shape = get_tf_node_attr(node, "shape")
        if not shape.unknown_rank:
            dims = [int(d.size) for d in shape.dim]
    except:  # pylint: disable=bare-except
        pass
    return dims


def get_tf_tensor_shape(tensor):
    shape = []
    try:
        shape = tensor.get_shape().as_list()
    except Exception:  # pylint: disable=broad-except
        shape = None
    return shape


def map_tf_dtype(dtype):
    if dtype:
        dtype = TF_TO_ONNX_DTYPE[dtype]
    return dtype


def map_numpy_to_onnx_dtype(np_dtype):
    for onnx_dtype, numpy_dtype in ONNX_TO_NUMPY_DTYPE.items():
        if numpy_dtype == np_dtype:
            return onnx_dtype
    raise ValueError("unsupported dtype " + np_dtype + " for mapping")


def map_onnx_to_numpy_type(onnx_type):
    return ONNX_TO_NUMPY_DTYPE[onnx_type]


def node_name(name):
    """Get node name without io#."""
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


def make_onnx_shape(shape):
    """shape with -1 is not valid in onnx ... make it a name."""
    if shape:
        # don't do this if input is a scalar
        return [make_name("unk") if i == -1 else i for i in shape]
    return shape


def port_name(name, nr=0):
    """Map node output number to name."""
    return name + ":" + str(nr)


def make_onnx_inputs_outputs(name, elem_type, shape, **kwargs):
    """Wrapper for creating onnx graph inputs or outputs
       name,  # type: Text
       elem_type,  # type: TensorProto.DataType
       shape,  # type: Optional[Sequence[int]]
    """
    if elem_type is None:
        elem_type = onnx_pb.TensorProto.UNDEFINED
    return helper.make_tensor_value_info(
        name,
        elem_type,
        make_onnx_shape(shape),
        **kwargs
    )


def find_opset(opset):
    """Find opset."""
    if opset is None or opset == 0:
        opset = defs.onnx_opset_version()
        if opset > constants.PREFERRED_OPSET:
            # if we use a newer onnx opset than most runtimes support, default to the one most supported
            opset = constants.PREFERRED_OPSET
    return opset


def get_of_node_attr(node, name):
    assert node.WhichOneof("op_type") == 'user_conf'
    attr_msg = node.user_conf.attr[name]
    attr_type = attr_msg.WhichOneof("value")
    #TODO(daquexian): a better check
    if attr_type == 'at_shape':
        return list(getattr(attr_msg, attr_type).dim)
    elif attr_type[:7] == 'at_list':
        return list(getattr(attr_msg, attr_type).val)
    else:
        return getattr(attr_msg, attr_type)


def get_tf_node_attr(node, name):
    """Parser TF node attribute."""
    if six.PY2:
        # For python2, TF get_attr does not accept unicode
        name = str(name)
    return node.get_attr(name)


def save_onnx_model(save_path_root, onnx_file_name, feed_dict, model_proto, include_test_data=False, as_text=False):
    """Save onnx model as file. Save a pbtxt file as well if as_text is True"""
    save_path = save_path_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if include_test_data:
        data_path = os.path.join(save_path, "test_data_set_0")
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        i = 0
        for data_key in feed_dict:
            data = feed_dict[data_key]
            t = numpy_helper.from_array(data)
            t.name = data_key
            data_full_path = os.path.join(data_path, "input_" + str(i) + ".pb")
            save_protobuf(data_full_path, t)
            i += 1

    target_path = os.path.join(save_path, onnx_file_name + ".onnx")
    save_protobuf(target_path, model_proto)
    if as_text:
        save_protobuf(target_path + ".pbtxt", model_proto, as_text=True)
    return target_path


def make_sure(bool_val, error_msg, *args):
    if not bool_val:
        raise ValueError("make_sure failure: " + error_msg % args)


def construct_graph_from_nodes(parent_g, nodes, outputs, shapes, dtypes):
    """Construct Graph from nodes and outputs with specified shapes and dtypes."""
    # pylint: disable=protected-access
    g = parent_g.create_new_graph_with_same_config()
    g.parent_graph = parent_g
    nodes = set(nodes)
    all_outputs = set()
    for op in nodes:
        all_outputs |= set(op.output)

        new_node = g.make_node(op.type, op.input, outputs=op.output, attr=op.attr, name=op.name,
                               skip_conversion=op.skip_conversion, infer_shape_dtype=False)
        body_graphs = op.graph.contained_graphs.pop(op.name, None)
        if body_graphs:
            for attr_name, body_graph in body_graphs.items():
                body_graph.parent_graph = g
                new_node.set_body_graph_as_attr(attr_name, body_graph)

    for i in all_outputs:
        if i not in g._output_shapes:
            g._output_shapes[i] = parent_g._output_shapes[i]
        if i not in g._dtypes:
            g._dtypes[i] = parent_g._dtypes[i]

    # handle cell graph: insert identity node, since sometimes we need output same output_id
    # as state_output and scan_out, but ONNX don't allow the same output_id to appear more
    # than once as output node.
    new_output_names = []
    for output, shape, dtype in zip(outputs, shapes, dtypes):
        node = g.make_node("Identity", inputs=[output], op_name_scope="sub_graph_ending_node",
                           shapes=[shape], dtypes=[dtype], infer_shape_dtype=False)
        new_output_names.append(node.output[0])
    g.outputs = new_output_names
    return g


def tf_name_scope(name):
    return '/'.join(name.split('/')[:-1])


def get_temp_directory():
    return os.environ.get("TF2ONNX_TEMP_DIRECTORY", tempfile.mkdtemp())


def delete_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_protobuf(path, message, as_text=False):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(message))
    else:
        with open(path, "wb") as f:
            f.write(message.SerializeToString())


def is_list_or_tuple(obj):
    return isinstance(obj, (list, tuple))


def is_unknown_dimension(dim):
    """  Return true if dim is not a positive integer value. """
    if dim is None or not isinstance(dim, int):
        return True
    return dim <= 0


def merge_shapes(shape1, shape2):
    """
    Merge 2 shapes, return merged shape, choose more specific dimension value from either side.
    Raise exception for mismatch.
    """
    if shape1 is None:
        return shape2
    if shape2 is None:
        return shape1

    make_sure(is_list_or_tuple(shape1), "invalid type for shape1")
    make_sure(is_list_or_tuple(shape2), "invalid type for shape2")
    make_sure(len(shape1) == len(shape2), "shapes rank mismatch: shape1=%s, shape2=%s", shape1, shape2)

    merged = []
    for d1, d2 in zip(shape1, shape2):
        d = d1
        if is_unknown_dimension(d1):
            d = d2
        elif not is_unknown_dimension(d2):
            make_sure(d1 == d2, "shapes dimension mismatch: shape1=%s, shape2=%s", shape1, shape2)
        merged.append(d)
    return merged


def are_shapes_compatible(src, dest):
    """
    Returns True iff src is compatible with dest.
    None is compatible with all shapes, different ranks are not considered as compatible
    """
    try:
        merge_shapes(src, dest)
        return True
    except:  # pylint: disable=bare-except
        return False


def are_shapes_equal(src, dest):
    """ Check whether 2 shapes are equal. """
    if src is None:
        return dest is None
    if dest is None:
        return src is None

    make_sure(is_list_or_tuple(src), "invalid type for src")
    make_sure(is_list_or_tuple(dest), "invalid type for dest")

    if len(src) != len(dest):
        return False
    return all(i == j for i, j in zip(src, dest))


def create_vague_shape_like(shape):
    make_sure(len(shape) >= 0, "rank should be >= 0")
    return [-1 for i in enumerate(shape)]


def get_onnx_version():
    return onnx.__version__


def get_tf_version():
    return LooseVersion(tf.__version__)


def make_opsetid(domain, version):
    make_sure(isinstance(version, int), "version must be an integer")
    return helper.make_opsetid(domain, version)


def is_onnx_domain(domain):
    if domain is None or domain == "":
        return True
    return False


def parse_bool(val):
    if val is None:
        return False
    return val.lower() in ("yes", "true", "t", "y", "1")


_is_debug_mode = parse_bool(os.environ.get(constants.ENV_TF2ONNX_DEBUG_MODE))


def is_debug_mode():
    return _is_debug_mode


def set_debug_mode(enabled):
    global _is_debug_mode
    _is_debug_mode = enabled


def get_max_value(np_dtype):
    return np.iinfo(np_dtype).max


def get_min_value(np_dtype):
    return np.iinfo(np_dtype).min


def get_url(url, path, max_retries=5):
    """ Download url and save to path. """
    retries = Retry(total=max_retries, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    response = session.get(url, allow_redirects=True)
    if response.status_code not in [200]:
        response.raise_for_status()

    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "wb") as f:
        f.write(response.content)


def have_same_inference_value(g, output_1, output_2):
    """
    If two outputs have the same value in inference.
    Check whether they come from the same subgraph and the same subgraphs
    contain nodes with the same attributes and share the same ancestors.
    """

    def is_same(node_1, node_2):
        # go further util two instance isn't the same
        if node_1 == node_2:
            return True
        # check body graph
        if node_1.get_body_graphs() or node_2.get_body_graphs():
            logger.warning("Comparing two nodes containing body graph isn't supported.")
            return False
        # check domain
        if node_1.domain != node_2.domain:
            return False
        # check type
        if node_1.type != node_2.type:
            return False
        # check onnx attributes
        if node_1.attr_onnx.keys() != node_2.attr_onnx.keys():
            return False
        for name in node_1.attr_onnx.keys(): # pylint: disable=consider-iterating-dictionary
            if node_1.get_attr_value(name) != node_2.get_attr_value(name):
                return False
        return True

    if output_1 == output_2:
        return True
    node_1 = g.get_node_by_output(output_1)
    node_2 = g.get_node_by_output(output_2)
    # compare their domain, attr, etc. see __eq__ in Node class
    if not is_same(node_1, node_2):
        return False

    for inp_1, inp_2 in zip(node_1.input, node_2.input):
        if not have_same_inference_value(g, inp_1, inp_2):
            return False
    return True


def is_tf_reverse_op(op):
    return op.type in ("ReverseV2", "ReverseSequence")


def is_tf_concat_op(op):
    return op.type in ("Concat", "ConcatV2", "ConcatV3")


def is_tf_tensor_array_gather_op(op):
    return op.type in ("TensorArrayGatherV2", "TensorArrayGatherV3")


def is_tf_tensor_array_write_op(op):
    return op.type in ("TensorArrayWriteV2", "TensorArrayWriteV3")


def is_tf_tensor_array_op(op):
    return op.type in ("TensorArrayV2", "TensorArrayV3")


def is_tf_loopcond_op(op):
    return op.type == "LoopCond"


def is_tf_select_op(op):
    return op.type == "Select"


def is_tf_slice_op(op):
    return op.type == "Slice"


def is_tf_const_op(op):
    return op.type in ["Const", "ConstV2"]
