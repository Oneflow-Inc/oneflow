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
import tensorflow as tf

from oneflow.python.onnx.load.common.tf_helper import tf_shape
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from .gather_and_scatter_mixin import GatherAndScatterMixin


@onnx_op("ScatterElements")
class ScatterElements(GatherAndScatterMixin, BackendHandler):
    @classmethod
    def version_11(cls, node, **kwargs):
        axis = node.attrs.get("axis", 0)
        data = kwargs["tensor_dict"][node.inputs[0]]
        indices = kwargs["tensor_dict"][node.inputs[1]]
        updates = kwargs["tensor_dict"][node.inputs[2]]

        # poocess negative axis
        axis = axis if axis >= 0 else tf.add(tf.rank(data), axis)

        # check are there any indices are out of bounds
        result = cls.chk_idx_out_of_bounds_along_axis(data, axis, indices)
        msg = "ScatterElements indices are out of bounds, please double check the indices and retry."
        with tf.control_dependencies(
            [tf.compat.v1.assert_equal(result, True, message=msg)]
        ):
            # process negative indices
            indices = cls.process_neg_idx_along_axis(data, axis, indices)

            # Calculate shape of the tensorflow version of indices tensor.
            sparsified_dense_idx_shape = tf_shape(updates)

            # Move on to convert ONNX indices to tensorflow indices in 2 steps:
            #
            # Step 1:
            #   What would the index tensors look like if updates are all
            #   dense? In other words, produce a coordinate tensor for updates:
            #
            #   coordinate[i, j, k ...] = [i, j, k ...]
            #   where the shape of "coordinate" tensor is same as that of updates.
            #
            # Step 2:
            #   But the coordinate tensor needs some correction because coord
            #   vector at position axis is wrong (since we assumed update is dense,
            #   but it is not at the axis specified).
            #   So we update coordinate vector tensor elements at psotion=axis with
            #   the sparse coordinate indices.

            idx_tensors_per_axis = tf.meshgrid(
                *list(
                    map(
                        lambda x: tf.range(x, dtype=tf.dtypes.int64),
                        sparsified_dense_idx_shape,
                    )
                ),
                indexing="ij"
            )
            idx_tensors_per_axis[axis] = indices
            dim_expanded_idx_tensors_per_axis = list(
                map(lambda x: tf.expand_dims(x, axis=-1), idx_tensors_per_axis)
            )
            coordinate = tf.concat(dim_expanded_idx_tensors_per_axis, axis=-1)

            # Now the coordinate tensor is in the shape
            # [updates.shape, updates.rank]
            # we need it to flattened into the shape:
            # [product(updates.shape), updates.rank]
            indices = tf.reshape(coordinate, [-1, tf.rank(data)])
            updates = tf.reshape(updates, [-1])

            return [tf.tensor_scatter_nd_update(data, indices, updates)]
