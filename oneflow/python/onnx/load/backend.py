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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass

from onnx import defs
from onnx import numpy_helper
from onnx.backend.base import Backend
from onnx.backend.base import Device
from onnx.backend.base import namedtupledict
from onnx.helper import make_opsetid
import tensorflow as tf
import oneflow as flow

from oneflow.python.onnx import util
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.onnx.load.backend_rep import TensorflowRep
from oneflow.python.onnx.load.common import get_device_option
from oneflow.python.onnx.load.common import get_unique_suffix
from oneflow.python.onnx.load.common import supports_device as common_supports_device
from oneflow.python.onnx.load.common.handler_helper import get_all_backend_handlers
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.graph import Node as OnnxNode
import oneflow.python.onnx.load.common as common
import io
import tempfile
import os
import shutil
import numpy as np
import onnx
import torch


@oneflow_export("from_pytorch")
def from_pytorch(torch_model, inputs):
    if type(inputs) is not list:
        inputs = [inputs]
    torch_model = torch_model.to("cpu")
    f = io.BytesIO()
    input_names = ["x_{}".format(i) for i in range(len(inputs))]
    torch.onnx.export(
        torch_model,
        tuple([torch.zeros(ipt.shape) for ipt in inputs]),
        f,
        input_names=input_names,
        output_names=["y"],
        opset_version=12,
    )
    model_str = f.getvalue()
    with open("/home/dev/files/temp.onnx", "wb") as f:
        f.write(model_str)
    onnx_model = onnx.load_model_from_string(model_str)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = "/tmp/tmp2/"
        BackendHandler.WEIGHT_SAVE_DIR = tmpdirname
        for x in onnx_model.graph.initializer:
            dir_name = os.path.join(tmpdirname, x.name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(os.path.join(dir_name, "out"), "wb") as f:
                value = numpy_helper.to_array(x)
                f.write(value.tobytes())
        for node in onnx_model.graph.node:
            node = OnnxNode(node)
            if node.op_type == "Constant":
                attr_value = node.attrs["value"]
                value = numpy_helper.to_array(attr_value)
                # we do not support 0d tensor
                if len(value.shape) == 0:
                    value = np.reshape(value, (1,))
                dir_name = os.path.join(tmpdirname, node.output_tensors[0])
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                with open(os.path.join(dir_name, "out"), "wb") as f:
                    f.write(value.tobytes())

        def write_fake_data(var_name, value):
            dir_name = os.path.join(tmpdirname, var_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(os.path.join(dir_name, "out"), "wb") as f:
                f.write(value.tobytes())

        train_step_name = "System-Train-TrainStep-temp_job"
        write_fake_data(train_step_name, np.array([0]))
        write_fake_data("v1", np.array([0], dtype=np.float32))

        d = prepare(onnx_model, blob_dict=dict(zip(input_names, inputs)))
        return d["y"]


class TensorflowBackend(Backend):
    """ Tensorflow Backend for ONNX
    """

    @classmethod
    def prepare(
        cls,
        model,
        device="CPU",
        strict=True,
        logging_level="INFO",
        blob_dict=None,
        **kwargs
    ):
        """Prepare an ONNX model for Tensorflow Backend.

    This function converts an ONNX model to an internel representation
    of the computational graph called TensorflowRep and returns
    the converted representation.

    :param model: The ONNX model to be converted.
    :param device: The device to execute this model on.
    :param strict: Whether to enforce semantic equivalence between the original model
      and the converted tensorflow model, defaults to True (yes, enforce semantic equivalence).
      Changing to False is strongly discouraged.
      Currently, the strict flag only affects the behavior of MaxPool and AveragePool ops.
    :param logging_level: The logging level, default is INFO. Change it to DEBUG
      to see more conversion details or to WARNING to see less

    :returns: A TensorflowRep class object representing the ONNX model
    """
        super(TensorflowBackend, cls).prepare(model, device, **kwargs)
        common.logger.setLevel(logging_level)
        common.logger.handlers[0].setLevel(logging_level)

        return cls.onnx_model_to_tensorflow_rep(model, strict, blob_dict=blob_dict)

    @classmethod
    def onnx_model_to_tensorflow_rep(cls, model, strict, blob_dict=None):
        """ Convert ONNX model to TensorflowRep.

    :param model: ONNX ModelProto object.
    :param strict: whether to enforce semantic equivalence between the original model
      and the converted tensorflow model.
    :return: TensorflowRep object.
    """

        # Models with IR_VERSION less than 3 does not have opset_import set.
        # We default to minimum opset, this behavior is consistent with
        # onnx checker.
        # c.f. https://github.com/onnx/onnx/blob/427ac0c1b792363d373e3d7e4eef97fa46458420/onnx/checker.cc#L478
        if model.ir_version < 3:
            opset_import = [make_opsetid(defs.ONNX_DOMAIN, 1)]
        else:
            opset_import = model.opset_import
        return cls._onnx_graph_to_tensorflow_rep(
            model.graph, opset_import, strict, blob_dict=blob_dict
        )

    @classmethod
    def _onnx_graph_to_tensorflow_rep(cls, graph_def, opset, strict, blob_dict=None):
        """ Convert ONNX graph to TensorflowRep.

        :param graph_def: ONNX GraphProto object.
        :param opset: ONNX OperatorSetIdProto list.
        :param strict: whether to enforce semantic equivalence between the original model
          and the converted tensorflow model.
        :param blob_dict: {name: oneflow_blob}, the inputs of onnx graph will be populated with oneflow_blob with the same name
        :return: TensorflowRep object.
        """
        if blob_dict is None:
            blob_dict = {}
        handlers = cls._get_handlers(opset)

        tf_rep_graph = tf.Graph()
        with tf_rep_graph.as_default():
            # initializer: TensorProtos representing the values to initialize
            # a given tensor.
            # initialized: A list of names of the initialized tensors.
            if graph_def.initializer:
                input_dict_items = cls._onnx_initializer_to_input_dict_items(
                    graph_def.initializer
                )
                initialized = {init.name for init in graph_def.initializer}
            else:
                input_dict_items = []
                initialized = set()

            # creating placeholders for currently unknown inputs
            for value_info in graph_def.input:
                if value_info.name in initialized:
                    continue
                shape = list(
                    d.dim_value if (d.dim_value > 0 and d.dim_param == "") else None
                    for d in value_info.type.tensor_type.shape.dim
                )
                if value_info.name not in blob_dict:
                    raise NotImplementedError(
                        "no blob named {}".format(value_info.name)
                    )
                # x = tf.compat.v1.placeholder(
                #     data_type.onnx2tf(value_info.type.tensor_type.elem_type),
                #     name=value_info_name,
                #     shape=shape,
                # )
                input_dict_items.append((value_info.name, blob_dict[value_info.name]))

            # tensor dict: this dictionary is a map from variable names
            # to the latest produced TF tensors of the given name.
            # This dictionary will get updated as we build the graph to
            # record the names of newly produced tensors.
            tensor_dict = dict(input_dict_items)
            # Since tensor dict may be updated, we need to keep a copy
            # of the original input dict where we track the earliest
            # defined tensors so we can have access to the placeholders
            # to feed in input tensors when we run the graph.
            input_dict = dict(input_dict_items)

            for node in graph_def.node:
                onnx_node = OnnxNode(node)
                output_ops = cls._onnx_node_to_tensorflow_op(
                    onnx_node, tensor_dict, handlers, opset=opset, strict=strict
                )
                curr_node_output_map = dict(zip(onnx_node.output_tensors, output_ops))
                tensor_dict.update(curr_node_output_map)
        return tensor_dict

    @classmethod
    def _onnx_initializer_to_input_dict_items(cls, initializer):
        """ Convert ONNX graph initializer to input dict items.

    :param initializer: ONNX graph initializer, list of TensorProto.
    :return: List of input dict items.
    """

        def tensor2list(onnx_tensor):
            # Use the onnx.numpy_helper because the data may be raw
            return numpy_helper.to_array(onnx_tensor).flatten().tolist()

        def validate_initializer_name(name):
            # Replace ":" with "_tf_" and append a unique suffix for
            # traceability
            return (
                name.replace(":", "_tf_") + "_" + get_unique_suffix()
                if ":" in name
                else name
            )

        def get_flow_shape(shape):
            if len(shape) == 0:
                return (1,)
            return shape

        return [
            (
                init.name,
                flow.get_variable(
                    name=validate_initializer_name(init.name),
                    shape=get_flow_shape(list(init.dims)),
                    initializer=flow.zeros_initializer(),
                    trainable=True,
                    dtype=util.Onnx2FlowDtype(init.data_type),
                )
                # tf.constant(
                #     tensor2list(init),
                #     shape=init.dims,
                #     dtype=data_type.onnx2tf(init.data_type),
                #     name=validate_initializer_name(init.name),
                # ),
            )
            for init in initializer
        ]

    @classmethod
    def _onnx_node_to_tensorflow_op(
        cls, node, tensor_dict, handlers=None, opset=None, strict=True
    ):
        """
    Convert onnx node to tensorflow op.

    Args:
      node: Onnx node object.
      tensor_dict: Tensor dict of graph.
      opset: Opset version of the operator set. Default 0 means using latest version.
      strict: whether to enforce semantic equivalence between the original model
        and the converted tensorflow model, defaults to True (yes, enforce semantic equivalence).
        Changing to False is strongly discouraged.
    Returns:
      Tensorflow op
    """
        handlers = handlers or cls._get_handlers(opset)
        handler = handlers[node.domain].get(node.op_type, None)
        if handler:
            output = handler.handle(node, tensor_dict=tensor_dict, strict=strict)
            if not isinstance(output, (list, tuple)):
                output = [output]
            return output
        else:
            raise ValueError("{} is not supported".format(node.op_type))

    @classmethod
    def _get_handlers(cls, opset):
        """ Get all backend handlers with opset.

    :param opset: ONNX OperatorSetIdProto list.
    :return: All backend handlers.
    """
        opset = opset or [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
        opset_dict = dict([(o.domain, o.version) for o in opset])
        return get_all_backend_handlers(opset_dict)

    @classmethod
    def supports_device(cls, device):
        return common_supports_device(device)

    @classmethod
    def onnx_graph_to_tensorflow_rep(cls, graph_def, strict=True):
        """
    Converts ONNX graph to TensorflowRep
    Args:
      graph_def:        the ONNX graph to be converted
      strict:           whether to enforce semantic equivalence between the
                        original model and the converted tensorflow model,
                        defaults to True (yes, enforce semantic equivalence).
    Returns:
      TensorflowRep object.
    """
        # get the opset of the installed ONNX
        opset = [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
        return cls._onnx_graph_to_tensorflow_rep(graph_def, opset, strict)


prepare = TensorflowBackend.prepare

run_model = TensorflowBackend.run_model

supports_device = TensorflowBackend.supports_device

onnx_graph_to_tensorflow_rep = TensorflowBackend.onnx_graph_to_tensorflow_rep
