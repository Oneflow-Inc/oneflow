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
import oneflow as flow
from oneflow.python.onnx.load.handler import BackendHandler
from oneflow.python.onnx.load.handler import onnx_op
from oneflow.python.onnx.load.handler import flow_func
from oneflow.python.ops import math_unary_elementwise_ops
from oneflow.python.ops import math_binary_elementwise_ops
from oneflow.python.ops import math_ops
from oneflow.python.ops import array_ops
from oneflow.python.ops import linalg
from oneflow.python.onnx.load.handlers.common import ArithmeticMixin, BasicMathMixin
from oneflow.python.onnx import util as onnx_util


@onnx_op("Add")
@flow_func(math_ops.add)
class Add(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Sub")
@flow_func(math_ops.subtract)
class Sub(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Mul")
@flow_func(math_ops.multiply)
class Mul(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Div")
@flow_func(math_ops.divide)
class Div(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Pow")
@flow_func(math_binary_elementwise_ops.pow)
class Pow(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        y = tensor_dict[node.input_tensor_names[1]]
        if len(y.shape) > len(x.shape):
            x = math_ops.broadcast_to_compatible_with(x, [y])
        elif len(x.shape) > len(y.shape):
            y = math_ops.broadcast_to_compatible_with(y, [x])
        return math_binary_elementwise_ops.pow(x, y)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls.version_1(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls.version_1(node, tensor_dict, **kwargs)


@onnx_op("Tanh")
@flow_func(math_unary_elementwise_ops.tanh_v2)
class Tanh(BasicMathMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return [cls.run_onnx_node(node, tensor_dict, **kwargs)]

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return [cls.run_onnx_node(node, tensor_dict, **kwargs)]


@onnx_op("Sigmoid")
@flow_func(math_ops.sigmoid)
class Sigmoid(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Gemm")
class Gemm(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        y = tensor_dict[node.input_tensor_names[1]]

        if len(node.input_tensor_names) > 2:
            z = tensor_dict[node.input_tensor_names[2]]
        else:
            z = 0

        transA = False if node.attrs.get("transA", 0) == 0 else True
        transB = False if node.attrs.get("transB", 0) == 0 else True
        alpha = node.attrs.get("alpha", 1.0)
        beta = node.attrs.get("beta", 1.0)

        return [
            alpha * linalg.matmul(x, y, transpose_a=transA, transpose_b=transB)
            + beta * z
        ]

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("MatMul")
@flow_func(linalg.matmul)
class MatMul(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        y = tensor_dict[node.input_tensor_names[1]]
        if len(y.shape) > len(x.shape):
            broadcast_shape = y.shape[:-2] + x.shape[-2:]
            constant_for_broadcast = flow.constant(
                value=0, dtype=flow.float32, shape=broadcast_shape
            )
            x = math_ops.broadcast_to_compatible_with(x, [constant_for_broadcast])
        elif len(x.shape) > len(y.shape):
            broadcast_shape = x.shape[:-2] + y.shape[-2:]
            constant_for_broadcast = flow.constant(
                value=0, dtype=flow.float32, shape=broadcast_shape
            )
            y = math_ops.broadcast_to_compatible_with(y, [constant_for_broadcast])
        return cls.run_onnx_node(node, tensor_dict, inputs=(x, y), **kwargs)


@onnx_op("Clip")
class Clip(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        x_dtype = x.dtype
        if cls.SINCE_VERSION < 11:
            # min/max were required and passed as attributes
            clip_value_min = node.attrs.get("min", None)
            clip_value_max = node.attrs.get("max", None)
        else:
            # min/max are optional and passed as input_tensor_names
            init_dict = kwargs["init_dict"]
            clip_value_min = (
                init_dict[node.input_tensor_names[1]].item()
                if len(node.input_tensor_names) > 1 and node.input_tensor_names[1] != ""
                else None
            )
            clip_value_max = (
                init_dict[node.input_tensor_names[2]].item()
                if len(node.input_tensor_names) > 2 and node.input_tensor_names[2] != ""
                else None
            )

        y = math_ops.clip_by_value(x, clip_value_min, clip_value_max)

        return y

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Sqrt")
@flow_func(math_unary_elementwise_ops.sqrt)
class Sqrt(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Erf")
@flow_func(math_unary_elementwise_ops.erf)
class Erf(BackendHandler):
    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Cast")
@flow_func(math_ops.cast)
class Cast(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        dtype = onnx_util.Onnx2FlowDtype(node.attrs.pop("to"))
        node.attrs["dtype"] = dtype
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
