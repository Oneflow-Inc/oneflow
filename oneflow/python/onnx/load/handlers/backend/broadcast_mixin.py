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
import tensorflow as tf


class BroadcastMixin(object):
    @classmethod
    def explicit_broadcast(cls, inputs, axis=None, tensor_dict=None):
        x = inputs[0] if isinstance(inputs[0], tf.Tensor) else tensor_dict[inputs[0]]
        y = inputs[1] if isinstance(inputs[1], tf.Tensor) else tensor_dict[inputs[1]]

        if np.prod(y.shape) == 1:
            return y

        if not isinstance(x, tf.Tensor) or not isinstance(y, tf.Tensor):
            raise ValueError("Targets for explicit broadcasting need to be Tensor.")

        if axis is None:
            return y

        total_num_dim = len(x.get_shape())
        if axis < 0:
            axis += total_num_dim

        if axis + len(y.get_shape()) == total_num_dim:
            return y

        dims = [axis + i for i in range(len(y.get_shape()))]
        new_y = y
        for i in range(total_num_dim):
            if i not in dims:
                new_y = tf.expand_dims(new_y, i)
        return new_y

    @classmethod
    def limited_broadcast(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        y = tensor_dict[node.inputs[1]]
        if node.attrs.get("broadcast") == 1:
            y = cls.explicit_broadcast([x, y], node.attrs.get("axis", None))
            return [cls.make_tensor_from_onnx_node(node, inputs=[x, y], **kwargs)]
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]
