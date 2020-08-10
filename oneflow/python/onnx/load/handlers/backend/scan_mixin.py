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
from onnx.helper import make_opsetid
import oneflow.python.onnx.load
from oneflow.python.onnx.load.common import data_type


class ScanMixin(object):
    @classmethod
    def scan(cls, node, input_dict, strict):
        current_opset = [make_opsetid(cls.DOMAIN, cls.VERSION)]

        body = node.attrs["body"]

        # in version 8, node.inputs[0] is the sequence_lens
        node_inputs = node.inputs if cls.SINCE_VERSION != 8 else node.inputs[1:]
        # M
        num_scan_inputs = int(node.attrs["num_scan_inputs"])
        # N = num_inputs - M
        num_state_vars = len(node_inputs) - num_scan_inputs
        # K = num_outputs - N
        num_scan_outputs = len(node.outputs) - num_state_vars

        """
        Function to run subgraph used with tf.scan
    """

        def run_subgraph(a, b):
            input_values = {}
            # set the input values for the subgraph
            # set the values for the state variables
            for i in range(num_state_vars):
                input_values[body.input[i].name] = a[i]
            # set the values for the scan inputs
            for i in range(num_scan_inputs):
                input_values[body.input[i + num_state_vars].name] = b[i]

            # get the tensor operations for the onnx graph
            subgraph_tensor_dict = oneflow.python.onnx.load.backend.onnx_graph_to_tensorflow_ops(
                subgraph=body,
                input_values=input_values,
                tensor_dict=input_dict,
                opset=current_opset,
                strict=strict,
            )
            # return sequence of tensors for every subgraph output
            outputs = [subgraph_tensor_dict[output.name] for output in body.output]
            return outputs

        scan_input_axes = node.attrs.get("scan_input_axes", [0] * num_scan_inputs)
        scan_input_directions = node.attrs.get(
            "directions" if cls.SINCE_VERSION == 8 else "scan_input_directions",
            [0] * num_scan_inputs,
        )
        scan_output_axes = node.attrs.get("scan_output_axes", [0] * num_scan_outputs)
        scan_output_directions = node.attrs.get(
            "scan_output_directions", [0] * num_scan_outputs
        )

        # if version 8 read the sequnce_lens from the first input
        if cls.SINCE_VERSION == 8:
            sequence_lens = input_dict[node.inputs[0]] if node.inputs[0] != "" else None

        inputs = [input_dict[node_input] for node_input in node_inputs]

        scan_inputs = inputs[num_state_vars:]
        # loop over all the scan inputs and apply transpose depending
        # on input axes provided and also reverse the scan inputs if
        # reverse direction for scan is provided
        for i in range(num_scan_inputs):
            # if input axes are different than 0, use transpose to scan over
            # the provided axes
            if scan_input_axes[i] != 0:
                transpose_perm = cls._calc_transpose_perm_input(
                    tf.rank(scan_inputs[i]), scan_input_axes[i]
                )
                scan_inputs[i] = tf.transpose(scan_inputs[i], transpose_perm)

            # check for reverse direction scans
            if scan_input_directions[i] == 1:
                # version 8 has a batch dimension
                axis = 0 if cls.SINCE_VERSION != 8 else 1
                scan_inputs[i] = tf.reverse(scan_inputs[i], [axis])

        state_vars_init = inputs[:num_state_vars]

        scan_outputs_init = []
        # generate sequence of zero tensors for all scan outputs
        # with the correct shape and dtype
        for scan_output in body.output[num_state_vars:]:
            tensor_type = scan_output.type.tensor_type
            shape = [
                d.dim_value if (d.dim_value > 0 and d.dim_param == "") else None
                for d in tensor_type.shape.dim
            ]
            dtype = data_type.onnx2tf(tensor_type.elem_type)
            scan_outputs_init.append(tf.zeros(shape, dtype=dtype))

        # tf.scan initilizer is state_variables_init + scan_outputs_init
        initializer = state_vars_init + scan_outputs_init

        if cls.SINCE_VERSION == 8:
            # version == 8
            # function to process the batches. it is used with tf.map_fn
            def run_batches(x):
                # state vars initial values per batch
                initial = x[0]
                # scan inputs per batch
                scan_inputs = x[1]
                # sequence length for the batch
                seq_len = x[2]

                # slice the input to the current sequence len
                scan_inputs = [scan_input[:seq_len, ...] for scan_input in scan_inputs]

                # run scan on the current batch
                out = tf.scan(
                    run_subgraph, scan_inputs, initializer=initial + scan_outputs_init
                )

                # pad to the original shape with zeros
                paddings = [[0, tf.shape(x[1][0], out_type=seq_len.dtype)[0] - seq_len]]
                for i in range(len(out)):
                    pads = tf.concat(
                        [
                            paddings,
                            tf.zeros([(tf.rank(out[i]) - 1), 2], dtype=tf.int32),
                        ],
                        axis=0,
                    )
                    out[i] = tf.pad(out[i], pads)
                return out

            if sequence_lens is None:
                # if sequence_lens is None, fill it with the shape of
                # the input axis 1
                sequence_lens = tf.fill(
                    [tf.shape(scan_inputs[0])[0]],
                    tf.shape(scan_inputs[0], out_type=tf.int32)[1],
                )

            output_types = [
                data_type.onnx2tf(output.type.tensor_type.elem_type)
                for output in body.output
            ]
            # run scan for every batch
            out = tf.map_fn(
                run_batches,
                (state_vars_init, scan_inputs, sequence_lens),
                dtype=output_types,
            )

            state_vars_outputs = []
            # extract the final values of the state variables
            for state_var in out[:num_state_vars]:
                state_vars_outputs.append(
                    tf.map_fn(
                        lambda x: x[0][x[1] - 1],
                        (state_var, sequence_lens),
                        state_var.dtype,
                    )
                )
        else:
            # version > 8
            # run the scan
            out = tf.scan(run_subgraph, scan_inputs, initializer=initializer)

            # extract the final values of the state variables
            state_vars_outputs = [
                state_var[tf.shape(state_var)[0] - 1]
                for state_var in out[:num_state_vars]
            ]

        scan_outputs = out[num_state_vars:]

        # post process the scan outputs depending on the directions and
        # axes provided.
        for i in range(num_scan_outputs):
            # check for reverse direction scan outputs
            if scan_output_directions[i] == 1:
                scan_outputs[i] = tf.reverse(scan_outputs[i], [0])

            if scan_output_axes[i] != 0:
                transpose_perm = cls._calc_transpose_perm_output(
                    tf.rank(scan_outputs[i]), scan_output_axes[i]
                )
                scan_outputs[i] = tf.transpose(scan_outputs[i], transpose_perm)

        return state_vars_outputs + scan_outputs

    @classmethod
    def _calc_transpose_perm_input(cls, rank, axis):
        if axis < 0:
            axis = rank + axis
        return tf.concat([[axis], tf.range(axis), tf.range(axis + 1, rank)], 0)

    @classmethod
    def _calc_transpose_perm_output(cls, rank, axis):
        if axis < 0:
            axis = rank + axis
        return tf.concat([tf.range(1, axis + 1), [0], tf.range(axis + 1, rank)], 0)
