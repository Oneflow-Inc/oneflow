# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Methods to load tensorflow graph from graphdef, checkpoint or saved_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from oneflow.python.onnx import utils

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument


def freeze_session(sess, keep_var_names=None, output_names=None, clear_devices=True):
    """Freezes the state of a session into a pruned computation graph."""
    output_names = [i.split(':')[:-1][0] for i in output_names]
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def(add_shapes=True)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(sess, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def remove_redundant_inputs(frozen_graph, input_names):
    """Remove redundant inputs not in frozen graph."""
    frozen_inputs = []
    # get inputs in frozen graph
    for n in frozen_graph.node:
        for inp in input_names:
            if utils.node_name(inp) == n.name:
                frozen_inputs.append(inp)
    deleted_inputs = list(set(input_names) - set(frozen_inputs))
    if deleted_inputs:
        logger.warning("inputs [%s] is not in frozen graph, delete them", ",".join(deleted_inputs))
    return frozen_inputs


def from_graphdef(model_path, input_names, output_names):
    """Load tensorflow graph from graphdef."""
    # make sure we start with clean default graph
    tf.reset_default_graph()
    with tf.Session() as sess:
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            frozen_graph = freeze_session(sess, output_names=output_names)
    input_names = remove_redundant_inputs(frozen_graph, input_names)
    # clean up
    tf.reset_default_graph()
    return frozen_graph, input_names, output_names


def from_checkpoint(model_path, input_names, output_names):
    """Load tensorflow graph from checkpoint."""
    # make sure we start with clean default graph
    tf.reset_default_graph()
    # model_path = checkpoint/checkpoint.meta
    saver = tf.train.import_meta_graph(model_path, clear_devices=True)
    with tf.Session() as sess:
        # restore from model_path minus the ".meta"
        saver.restore(sess, model_path[:-5])
        frozen_graph = freeze_session(sess, output_names=output_names)
    input_names = remove_redundant_inputs(frozen_graph, input_names)
    # clean up
    tf.reset_default_graph()
    return frozen_graph, input_names, output_names


def from_saved_model(model_path, input_names, output_names, signatures=None):
    """Load tensorflow graph from saved_model."""
    # make sure we start with clean default graph
    tf.reset_default_graph()
    inputs = {}
    outputs = {}
    try:
        # pylint: disable=C0415
        from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
        # pylint: disable=unnecessary-lambda
        get_signature_def = lambda meta_graph_def, k: \
            signature_def_utils.get_signature_def_by_key(meta_graph_def, k)
    except ImportError:
        # TF1.12 changed the api
        get_signature_def = lambda meta_graph_def, k: meta_graph_def.signature_def[k]

    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)

        if signatures is None:
            signatures = []
            for k in meta_graph_def.signature_def.keys():
                if k.startswith("_"):
                    # consider signatures starting with '_' private
                    continue
                signatures.append(k)
            if len(signatures) > 1:
                logger.warning("found multiple signatures %s in saved_model, pass --signature_def in command line",
                               signatures)
        for k in signatures:
            inputs_tensor_info = get_signature_def(meta_graph_def, k).inputs
            for _, input_tensor in sorted(inputs_tensor_info.items()):
                inputs[input_tensor.name] = sess.graph.get_tensor_by_name(input_tensor.name)
            outputs_tensor_info = get_signature_def(meta_graph_def, k).outputs
            for _, output_tensor in sorted(outputs_tensor_info.items()):
                outputs[output_tensor.name] = sess.graph.get_tensor_by_name(output_tensor.name)
        frozen_graph = freeze_session(sess, output_names=list(outputs.keys()))
    if input_names is None:
        input_names = inputs.keys()
    input_names = remove_redundant_inputs(frozen_graph, input_names)
    # clean up
    tf.reset_default_graph()
    return frozen_graph, input_names, list(outputs.keys())
