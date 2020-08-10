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

import tensorflow as tf

from onnx.backend.base import BackendRep, namedtupledict


class TensorflowRep(BackendRep):
    def __init__(self, graph=None, inputs=None, outputs=None, tensor_dict=None):
        super(TensorflowRep, self).__init__()
        self._graph = graph
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._tensor_dict = tensor_dict or {}

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @property
    def tensor_dict(self):
        return self._tensor_dict

    @tensor_dict.setter
    def tensor_dict(self, tensor_dict):
        self._tensor_dict = tensor_dict

    def run(self, inputs, **kwargs):
        """ Run TensorflowRep.

    :param inputs: Given inputs.
    :param kwargs: Other args.
    :return: Outputs.
    """
        super(TensorflowRep, self).run(inputs, **kwargs)

        # TODO: handle name scope if necessary
        with self.graph.as_default():
            with tf.compat.v1.Session() as sess:
                if isinstance(inputs, dict):
                    feed_dict = inputs
                elif isinstance(inputs, list) or isinstance(inputs, tuple):
                    if len(self.inputs) != len(inputs):
                        raise RuntimeError(
                            "Expected {} values for uninitialized "
                            "graph inputs ({}), but got {}.".format(
                                len(self.inputs), ", ".join(self.inputs), len(inputs)
                            )
                        )
                    feed_dict = dict(zip(self.inputs, inputs))
                else:
                    # single input
                    feed_dict = dict([(self.inputs[0], inputs)])

                feed_dict = {
                    self.tensor_dict[key]: feed_dict[key] for key in self.inputs
                }

                sess.run(tf.compat.v1.global_variables_initializer())
                outputs = [self.tensor_dict[output] for output in self.outputs]

                output_values = sess.run(outputs, feed_dict=feed_dict)
                return namedtupledict("Outputs", self.outputs)(*output_values)

    def export_graph(self, path):
        """Export backend representation to a Tensorflow proto file.

    This function obtains the graph proto corresponding to the ONNX
    model associated with the backend representation and serializes
    to a protobuf file.

    :param path: The path to the output TF protobuf file.

    :returns: none.
    """
        graph_proto = self.graph.as_graph_def()
        # rename the output nodes
        meaningful_names = {}
        for output_name in self.outputs:
            meaningful_names[
                self.tensor_dict[output_name].name.replace(":0", "")
            ] = output_name
        for node in graph_proto.node:
            if node.name in meaningful_names.keys():
                node.name = meaningful_names[node.name]

        file = open(path, "wb")
        file.write(graph_proto.SerializeToString())
        file.close()
