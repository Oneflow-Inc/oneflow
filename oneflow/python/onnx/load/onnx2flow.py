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
import oneflow as flow

from oneflow.python.onnx import util
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.onnx.load.handler import BackendHandler

from oneflow.python.onnx.load.handlers import *
from oneflow.python.onnx.onnx_wrapper import Node as OnnxNode
import io
import tempfile
import os
import shutil
import numpy as np
import onnx
import torch
import logging

try:
    import onnxsim

    has_onnxsim = True
except ImportError:
    has_onnxsim = False

logger = logging.getLogger(__name__)


@oneflow_export("from_onnx")
def from_onnx(
    onnx_model: onnx.ModelProto, inputs, model_weight_dir="/tmp/tmp", do_onnxsim=True
):
    input_names = [x.name for x in onnx_model.graph.input]
    if type(inputs) is not dict:
        assert (
            len(input_names) == 1
        ), "Please use input dict if the model has multiple inputs"
        inputs = {input_names[0]: inputs}
    if do_onnxsim and has_onnxsim:
        dict(zip(input_names, [x.shape for x in inputs.values()]))
        onnx_model, _ = onnxsim.simplify(
            onnx_model,
            skip_shape_inference=False,
            input_shapes=dict(zip(input_names, [x.shape for x in inputs.values()])),
        )
    elif do_onnxsim:
        logger.info(
            "We recommend installing onnx-simplifier so that OneFlow can remove the redundant ONNX nodes"
        )

    # if not os.path.exists("/home/zhangxiaoyu/temp_onnx"):
    #     os.makedirs("/home/zhangxiaoyu/temp_onnx")
    # onnx.save(onnx_model, "/home/zhangxiaoyu/temp_onnx/temp.onnx")

    if os.path.exists(model_weight_dir):
        shutil.rmtree(model_weight_dir)
    BackendHandler.WEIGHT_SAVE_DIR = model_weight_dir
    for x in onnx_model.graph.initializer:
        dir_name = os.path.join(model_weight_dir, x.name)
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
            dir_name = os.path.join(model_weight_dir, node.output_tensor_names[0])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(os.path.join(dir_name, "out"), "wb") as f:
                f.write(value.tobytes())

    def write_fake_data(var_name, value):
        dir_name = os.path.join(model_weight_dir, var_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(os.path.join(dir_name, "out"), "wb") as f:
            f.write(value.tobytes())

    train_step_name = "System-Train-TrainStep-temp_job"
    write_fake_data(train_step_name, np.array([0]))
    write_fake_data("v1", np.array([0], dtype=np.float32))

    d = prepare(onnx_model, blob_dict=inputs)
    output_names = [x.name for x in onnx_model.graph.output]
    if len(output_names) == 1:
        return d[output_names[0]]
    return {output_name: d[output_name] for output_name in output_names}


@oneflow_export("from_pytorch")
def from_pytorch(
    torch_model, inputs, model_weight_dir="/tmp", do_onnxsim=True, train_flag=True
):
    if type(inputs) is not list:
        inputs = [inputs]
    input_names = ["x_{}".format(i) for i in range(len(inputs))]

    torch_model = torch_model.to("cpu")

    f = io.BytesIO()
    torch.onnx.export(
        torch_model,
        tuple([torch.zeros(ipt.shape) for ipt in inputs]),
        f,
        input_names=input_names,
        opset_version=12,
        training=train_flag,
    )
    model_str = f.getvalue()
    onnx_model = onnx.load_model_from_string(model_str)
    return from_onnx(
        onnx_model,
        dict(zip(input_names, inputs)),
        model_weight_dir=model_weight_dir,
        do_onnxsim=do_onnxsim,
    )


def get_all_backend_handlers(opset_dict):
    """ Get a dict of all backend handler classes.
  e.g. {'domain': {'Abs': Abs handler class}, ...}, }.
  :param opset_dict: A dict of opset. e.g. {'domain': version, ...}
  :return: Dict.
  """
    handlers = {}
    for handler in BackendHandler.__subclasses__():
        handler.check_cls()

        domain = handler.DOMAIN
        version = opset_dict[domain]
        handler.VERSION = version

        since_version = 1
        if defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
            try:
                since_version = defs.get_schema(
                    handler.ONNX_OP,
                    domain=handler.DOMAIN,
                    max_inclusive_version=version,
                ).since_version
            except RuntimeError:
                logger.info(
                    "Fail to get since_version of {} in domain `{}` "
                    "with max_inclusive_version={}. Set to 1.".format(
                        handler.ONNX_OP, handler.DOMAIN, version
                    )
                )
        else:
            logger.info(
                "Unknown op {} in domain `{}`.".format(
                    handler.ONNX_OP, handler.DOMAIN or "ai.onnx"
                )
            )
        handler.SINCE_VERSION = since_version
        handlers.setdefault(domain, {})[handler.ONNX_OP] = handler
    return handlers


class OneflowBackend(Backend):
    """ Oneflow Backend for ONNX
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
        """Prepare an ONNX model for Oneflow Backend.
    :param model: The ONNX model to be converted.
    :param device: The device to execute this model on.
    :param strict: Whether to enforce semantic equivalence between the original model
      and the converted oneflow model, defaults to True (yes, enforce semantic equivalence).
      Changing to False is strongly discouraged.
      Currently, the strict flag only affects the behavior of MaxPool and AveragePool ops.
    :param logging_level: The logging level, default is INFO. Change it to DEBUG
      to see more conversion details or to WARNING to see less
    :returns: The variable dict of the converted oneflow model
    """
        super(OneflowBackend, cls).prepare(model, device, **kwargs)
        logger.setLevel(logging_level)

        return cls.onnx_model_to_oneflow(model, strict, blob_dict=blob_dict)

    @classmethod
    def onnx_model_to_oneflow(cls, model, strict, blob_dict=None):
        """ Convert ONNX model to oneflow.
    :param model: ONNX ModelProto object.
    :param strict: whether to enforce semantic equivalence between the original model
      and the converted oneflow model.
    :return: The variable dict of the converted oneflow model
    """

        # Models with IR_VERSION less than 3 does not have opset_import set.
        # We default to minimum opset, this behavior is consistent with
        # onnx checker.
        # c.f. https://github.com/onnx/onnx/blob/427ac0c1b792363d373e3d7e4eef97fa46458420/onnx/checker.cc#L478
        if model.ir_version < 3:
            opset_import = [make_opsetid(defs.ONNX_DOMAIN, 1)]
        else:
            opset_import = model.opset_import
        return cls._onnx_graph_to_oneflow(
            model.graph, opset_import, strict, blob_dict=blob_dict
        )

    @classmethod
    def _onnx_graph_to_oneflow(cls, graph_def, opset, strict, blob_dict=None):
        """ Convert ONNX graph to oneflow.
        :param graph_def: ONNX GraphProto object.
        :param opset: ONNX OperatorSetIdProto list.
        :param strict: whether to enforce semantic equivalence between the original model
          and the converted oneflow.
        :param blob_dict: {name: oneflow_blob}, the inputs of onnx graph will be populated with oneflow_blob with the same name
        :return: The variable dict of the converted oneflow model
        """
        if blob_dict is None:
            blob_dict = {}
        handlers = cls._get_handlers(opset)

        # initializer: TensorProtos representing the values to initialize
        # a given tensor.
        # initialized: A list of names of the initialized tensors.
        if graph_def.initializer:
            input_dict_items = cls._onnx_initializer_to_input_dict_items(
                graph_def.initializer
            )
            initialized = {
                init.name: onnx.numpy_helper.to_array(init)
                for init in graph_def.initializer
            }
        else:
            input_dict_items = []
            initialized = {}

        for node in graph_def.node:
            node = OnnxNode(node)
            if node.op_type == "Constant":
                initialized[node.output_tensor_names[0]] = numpy_helper.to_array(
                    node.attrs["value"]
                )

        # creating placeholders for currently unknown inputs
        for value_info in graph_def.input:
            if value_info.name in initialized:
                continue
            shape = list(
                d.dim_value if (d.dim_value > 0 and d.dim_param == "") else None
                for d in value_info.type.tensor_type.shape.dim
            )
            if value_info.name not in blob_dict:
                raise NotImplementedError("no blob named {}".format(value_info.name))
            input_dict_items.append((value_info.name, blob_dict[value_info.name]))

        # tensor dict: this dictionary is a map from variable names
        # to the latest produced oneflow variables of the given name.
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
            output_ops = cls._onnx_node_to_oneflow_op(
                onnx_node,
                tensor_dict,
                initialized,
                handlers,
                opset=opset,
                strict=strict,
            )
            curr_node_output_map = dict(zip(onnx_node.output_tensor_names, output_ops))
            tensor_dict.update(curr_node_output_map)
        return tensor_dict

    @classmethod
    def _onnx_initializer_to_input_dict_items(cls, initializer):
        """ Convert ONNX graph initializer to input dict items.
    :param initializer: ONNX graph initializer, list of TensorProto.
    :return: List of input dict items.
    """

        def get_flow_shape(shape):
            if len(shape) == 0:
                return (1,)
            return shape

        return [
            (
                init.name,
                flow.get_variable(
                    name=init.name,
                    shape=get_flow_shape(list(init.dims)),
                    initializer=flow.zeros_initializer(),
                    trainable=True,
                    dtype=util.Onnx2FlowDtype(init.data_type),
                ),
            )
            for init in initializer
        ]

    @classmethod
    def _onnx_node_to_oneflow_op(
        cls, node, tensor_dict, init_dict, handlers=None, opset=None, strict=True
    ):
        """
    Convert onnx node to oneflow op.
    Args:
      node: Onnx node object.
      tensor_dict: Tensor dict of graph.
      opset: Opset version of the operator set. Default 0 means using latest version.
      strict: whether to enforce semantic equivalence between the original model
        and the converted oneflow model, defaults to True (yes, enforce semantic equivalence).
        Changing to False is strongly discouraged.
    Returns:
      oneflow op
    """
        handlers = handlers or cls._get_handlers(opset)
        handler = handlers[node.domain].get(node.op_type, None)
        if handler:
            output = handler.handle(
                node, tensor_dict, init_dict=init_dict, strict=strict
            )
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


prepare = OneflowBackend.prepare
