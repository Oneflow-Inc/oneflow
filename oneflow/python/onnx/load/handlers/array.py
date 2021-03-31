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

import numpy as np
import oneflow as flow

from oneflow.python.onnx.load.handler import BackendHandler
from oneflow.python.onnx.load.handler import onnx_op
from oneflow.python.onnx.load.handler import flow_func
from oneflow.python.ops import array_ops
import oneflow.typing as tp


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
        init_dict = kwargs["init_dict"]
        x = tensor_dict[node.input_tensor_names[0]]
        if cls.SINCE_VERSION == 1:
            shape = node.attrs["shape"]
        else:  # since_version >= 5
            shape = init_dict[node.input_tensor_names[1]]
            node.attrs["shape"] = shape.tolist()
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


@onnx_op("Concat")
@flow_func(array_ops.concat)
class Concat(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        inputs = [tensor_dict[inp] for inp in node.input_tensor_names]
        return cls.run_onnx_node(node, tensor_dict, inputs=[inputs])

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_4(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Unsqueeze")
@flow_func(array_ops.expand_dims)
class Unsqueeze(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        axes = node.attrs.pop("axes")
        if len(axes) != 1:
            x = tensor_dict[node.input_tensor_names[0]]
            for axis in sorted(axes):
                x = array_ops.expand_dims(x, axis=axis)
            return x
        node.attrs["axis"] = axes[0]
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Squeeze")
@flow_func(array_ops.squeeze)
class Squeeze(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        axes = node.attrs.pop("axes")
        node.attrs["axis"] = axes
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

# TODO(BBuf) add expand op: https://github.com/Oneflow-Inc/oneflow/pull/4164
# This is a temporary solution of senet

@onnx_op("Expand")
@flow_func(array_ops.broadcast_like)
class Expand(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        import oneflow as flow

        x = tensor_dict[node.input_tensor_names[0]]
        init_dict = kwargs["init_dict"]
        shape = init_dict[node.input_tensor_names[1]].tolist()
        like_tensor = flow.constant(
            value=1.0,
            dtype=flow.float32,
            shape=(shape[0], shape[1], shape[2], shape[3]),
        )
        return array_ops.broadcast_like(x, like=like_tensor, broadcast_axes=(2, 3))

    @classmethod
    def version_8(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Transpose")
@flow_func(array_ops.transpose)
class Transpose(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Gather")
@flow_func(array_ops.gather)
class Gather(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        output = cls.run_onnx_node(node, tensor_dict, **kwargs)
        init_dict = kwargs["init_dict"]
        if node.input_tensor_names[1] not in init_dict:
            # TODO(daquexian): handle 0-d indices here
            return output
        else:
            if len(init_dict[node.input_tensor_names[1]].shape) == 0:
                output = array_ops.squeeze(output, axis=[node.attrs["axis"]])
            return output


@onnx_op("Slice")
@flow_func(array_ops.slice_v2)
class Slice(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        axes = node.attrs.pop("axes", list(range(len(x.shape))))
        ends = node.attrs.pop("ends")
        starts = node.attrs.pop("starts")
        slice_tup_list = []
        j = 0
        for i in range(len(x.shape)):
            if i in axes:
                slice_tup_list.append((starts[j], ends[j], 1))
                j = j + 1
            else:
                slice_tup_list.append((None, None, None))
        node.attrs["slice_tup_list"] = slice_tup_list
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_10(cls, node, tensor_dict, **kwargs):
        init_dict = kwargs["init_dict"]
        x = tensor_dict[node.input_tensor_names[0]]
        axes = list(range(len(x.shape)))
        if len(node.input_tensor_names) > 3:
            axes = init_dict.get(node.input_tensor_names[3], list(range(len(x.shape))))
        steps = [1] * len(x.shape)
        if len(node.input_tensor_names) > 4:
            steps = init_dict.get(node.input_tensor_names[4], [1] * len(x.shape))
        starts = init_dict[node.input_tensor_names[1]]
        ends = init_dict[node.input_tensor_names[2]]
        slice_tup_list = []
        j = 0
        for i in range(len(x.shape)):
            if i in axes:
                start, end, step = int(starts[j]), int(ends[j]), int(steps[j])
                if start == np.iinfo(np.int64).max:
                    start = None
                if end in [np.iinfo(np.int64).max, np.iinfo(np.int64).min]:
                    end = None
                slice_tup_list.append((start, end, step))
                j = j + 1
            else:
                slice_tup_list.append((None, None, None))
        node.attrs["slice_tup_list"] = slice_tup_list
        node.input_tensor_names = node.input_tensor_names[:1]
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls.version_10(node, tensor_dict, **kwargs)


@onnx_op("Split")
class Split(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        axis = node.attrs.get("axis")
        split = node.attrs.get("split")
        index = 0
        ans = []
        for i in range(len(split)):
            if axis == 1:
                tmp = flow.experimental.logical_slice(
                    x,
                    [
                        [None, None, None],
                        [index, index + split[i], 1],
                        [None, None, None],
                        [None, None, None],
                    ],
                )
            elif axis == 3:
                tmp = flow.experimental.logical_slice(
                    x,
                    [
                        [None, None, None],
                        [None, None, None],
                        [None, None, None],
                        [index, index + split[i], 1],
                    ],
                )
            else:
                raise ValueError("axis != 0 or 3 is not supported")
            index += split[i]
            ans.append(tmp)
        return ans

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_2(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Min")
class Min(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        min_v = kwargs["init_dict"][node.input_tensor_names[1]]
        return flow.math.clip_by_value(x, min_value=min_v)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_8(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Max")
class Max(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        max_v = tensor_dict[node.input_tensor_names[1]]
        max_v = kwargs["init_dict"][node.input_tensor_names[1]]
        return flow.math.clip_by_value(x, max_value=max_v)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_8(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)