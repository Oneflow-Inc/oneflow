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
import operator
from functools import reduce

from oneflow.python.onnx.load.handler import BackendHandler
from oneflow.python.onnx.load.handler import onnx_op
from oneflow.python.onnx.load.handler import flow_func
from oneflow.python.ops import array_ops


@onnx_op("Identity")
@flow_func(array_ops.identity)
class Identity(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Reshape")
@flow_func(array_ops.reshape)
class Reshape(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        init_dict = kwargs['init_dict']
        x = tensor_dict[node.input_tensor_names[0]]
        if cls.SINCE_VERSION == 1:
          shape = node.attrs["shape"]
        else:  # since_version >= 5
          shape = init_dict[node.input_tensor_names[1]]
          node.attrs['shape'] = shape.tolist()
          del node.input_tensor_names[1]
        # TODO(daquexian)): update oneflow reshape to support 0 and np.ndarray
        return [cls.run_onnx_node(node, tensor_dict, **kwargs)]

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_5(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    
@onnx_op("Flatten")
class Flatten(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        shape = x.shape
        axis = node.attrs.get("axis", 1)

        if axis == 0:
            cal_shape = (1, -1)
        else:
            cal_shape = (
                reduce(operator.mul, shape[:axis], 1),
                reduce(operator.mul, shape[axis:]),
            )
            # cal_shape = (tf.reduce_prod(shape[0:axis]),
            # tf.reduce_prod(shape[axis:tf.size(shape)]))
        return array_ops.reshape(x, cal_shape)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    
