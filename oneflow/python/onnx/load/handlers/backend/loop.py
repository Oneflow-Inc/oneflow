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

import oneflow.python.onnx.load
from onnx.helper import make_opsetid
from oneflow.python.onnx.load.common import data_type
from oneflow.python.onnx.load.common import exception
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op


@onnx_op("Loop")
class Loop(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        body = node.attrs["body"]
        tensor_dict = kwargs["tensor_dict"]
        M = tensor_dict[node.inputs[0]] if node.inputs[0] != "" else None
        cond = (
            tf.cast(tensor_dict[node.inputs[1]], tf.bool)
            if node.inputs[1] != ""
            else None
        )
        v_initial = [tensor_dict[graph_input] for graph_input in node.inputs[2:]]
        v_shapes = [v.get_shape() for v in v_initial]
        current_opset = [make_opsetid(cls.DOMAIN, cls.VERSION)]
        # outputs of the body will be in this format:
        # (condition, loop carried dependencies..., scan_outputs...)
        scan_outputs_start_index = 1 + len(v_initial)
        scan_outputs = [
            tf.TensorArray(
                dtype=data_type.onnx2tf(body.output[i].type.tensor_type.elem_type),
                size=0,
                dynamic_size=True,
            )
            for i in range(scan_outputs_start_index, len(body.output))
        ]
        scan_outputs_shapes = [tf.TensorShape(None) for o in scan_outputs]

        def run_subgraph(cond, v, scan_outputs):
            input_values = {}
            input_values[body.input[0].name] = M
            input_values[body.input[1].name] = cond
            for i in range(2, len(body.input)):
                input_values[body.input[i].name] = v[i - 2]
            subgraph_tensor_dict = oneflow.python.onnx.load.backend.onnx_graph_to_tensorflow_ops(
                subgraph=body,
                input_values=input_values,
                tensor_dict=tensor_dict,
                opset=current_opset,
            )
            outputs = [subgraph_tensor_dict[output.name] for output in body.output]
            for i in range(scan_outputs_start_index, len(outputs)):
                s_index = i - scan_outputs_start_index
                insert_index = scan_outputs[s_index].size()
                scan_outputs[s_index] = scan_outputs[s_index].write(
                    insert_index, outputs[i]
                )
            return outputs[0], outputs[1:scan_outputs_start_index], scan_outputs

        # for loop
        if M is not None and cond is None:
            M = tf.cast(M, tf.int32)
            condition = lambda cond, v, scan_outputs: True
            _, v_final, scan_outputs = tf.while_loop(
                cond=condition,
                body=run_subgraph,
                loop_vars=["", v_initial, scan_outputs],
                shape_invariants=[tf.TensorShape(None), v_shapes, scan_outputs_shapes],
                maximum_iterations=M,
            )
        # while and do-while loop
        elif M is None and cond is not None:
            condition = lambda cond, v, scan_outputs: tf.reduce_all(
                tf.equal(cond, True)
            )
            cond, v_final, scan_outputs = tf.while_loop(
                cond=condition,
                body=run_subgraph,
                loop_vars=[cond, v_initial, scan_outputs],
                shape_invariants=[tf.TensorShape(None), v_shapes, scan_outputs_shapes],
            )
        # combine for loop and while loop together
        elif M is not None and cond is not None:
            M = tf.cast(M, tf.int32)
            condition = lambda cond, v, scan_outputs: tf.reduce_all(
                tf.equal(cond, True)
            )
            cond, v_final, scan_outputs = tf.while_loop(
                cond=condition,
                body=run_subgraph,
                loop_vars=[cond, v_initial, scan_outputs],
                shape_invariants=[tf.TensorShape(None), v_shapes, scan_outputs_shapes],
                maximum_iterations=M,
            )
        else:  # M is None and cond is None
            exception.OP_UNSUPPORTED_EXCEPT(
                "Both M and cond in Loop are not set at the same time",
                "Tensorflow.(PS. if you want to create a do-while loop "
                + "then please set cond to True or 1)",
            )

        scan_outputs_tensors = [o.stack() for o in scan_outputs]
        if scan_outputs_start_index == len(body.output):
            # there is no scan_output in the body graph
            return [v_final]
        else:
            return [v_final, scan_outputs_tensors]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
