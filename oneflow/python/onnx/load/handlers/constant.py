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
import numpy as np

from onnx import numpy_helper

import oneflow as flow
from oneflow.python.ops import get_variable
from oneflow.python.onnx import util
from oneflow.python.onnx.load.handler import BackendHandler
from oneflow.python.onnx.load.handler import onnx_op
from oneflow.python.onnx.load.handler import flow_func

import os


@onnx_op("Constant")
@flow_func(get_variable.api_get_variable)
class Constant(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        attr_value = node.attrs["value"]
        dtype = util.Onnx2FlowDtype(attr_value.data_type)
        shape = numpy_helper.to_array(attr_value).shape
        # we do not support 0d tensor
        if len(shape) == 0:
            shape = (1,)
        return [
            cls.run_onnx_node(
                node,
                tensor_dict,
                # inputs=[value],
                # attrs={"dtype": dtype}
                name=node.output_tensor_names[0],
                attrs={
                    "dtype": dtype,
                    "trainable": False,
                    "shape": shape,
                    "initializer": flow.zeros_initializer(),
                },
            )
        ]

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        # either value or sparse_value
        if "value" in node.attrs:
            return cls._common(node, tensor_dict, **kwargs)
        else:
            raise NotImplementedError("sparse tensor is not supported")

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        if "value" in node.attrs or "sparse_value" in node.attrs:
            return cls.version_11(node, tensor_dict, **kwargs)
        raise NotImplementedError("opset 12 constant is not supported")
