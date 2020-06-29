# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.oneflow.python.onnx - rewrite oneflow graph to onnx graph
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import sys
import traceback
import itertools
import logging

import numpy as np
from onnx import helper, onnx_pb

import oneflow
import oneflow.python.onnx
import oneflow.python.onnx.onnx_opset  # pylint: disable=unused-import
from oneflow.python.onnx.graph import Graph
from . import constants, schemas, util, handler, optimizer

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.session_context as session_ctx
import os
import os.path

logger = logging.getLogger(__name__)


def oneflow_to_onnx_naive(graph, shape_override):
    """
    Convert node from oneflow format to onnx format.
    Convert the oneflow nodes into an onnx graph with minimal rewrites so
    we can use the onnx graph as intermediate graph.
    The input/output/attr of each node are kept here and will be converted in other
    following functions.
    """
    dtypes = {}
    for lbn in graph.helper.lbn2logical_blob_desc:
        lbd = graph.helper.lbn2logical_blob_desc[lbn]
        if lbn not in shape_override:
            shape_override[lbn] = list(lbd.body.shape.dim)
        dtypes[lbn] = util.map_flow_dtype(lbd.body.data_type)

    # some stats
    op_cnt = collections.Counter()
    attr_cnt = collections.Counter()
    onnx_nodes = []

    def is_user_op(node):
        return node.WhichOneof("op_type") == "user_conf"

    def get_op_conf(node):
        conf_type = node.WhichOneof("op_type")
        conf = getattr(node, conf_type)
        return conf

    def get_op_type(node):
        if is_user_op(node):
            return node.user_conf.op_type_name
        return node.WhichOneof("op_type")[:-5]

    def get_inputs(node):
        if is_user_op(node):
            ibns = handler.flow_op.ibn4op_type(get_op_type(node))
            if ibns is None:
                return list(
                    itertools.chain(*[x.s for x in node.user_conf.input.values()])
                )
            ipts = []
            for ibn in ibns:
                for key, val in node.user_conf.input.items():
                    if key == ibn:
                        assert len(val.s) == 1
                        ipts.append(val.s[0])
                        break
                else:
                    raise ValueError(
                        "ibn {} of node {} (type {}) not found".format(
                            ibn, node.name, get_op_type(node)
                        )
                    )
            return ipts
        else:
            conf = get_op_conf(node)
            # it cannot cover all legacy op but it's enough
            if hasattr(conf, "in"):
                op_in = getattr(conf, "in")
                if isinstance(op_in, str):
                    return [op_in]
                else:
                    return op_in
            else:
                return []

    def get_outputs(node):
        if is_user_op(node):
            obns = handler.flow_op.obn4op_type(get_op_type(node))
            if obns is None:
                assert all([len(x.s) == 1 for x in node.user_conf.output.values()])
                return [x.s[0] for x in node.user_conf.output.values()]
            outputs = []
            for obn in obns:
                for key, val in node.user_conf.output.items():
                    if key == obn:
                        assert len(val.s) == 1
                        outputs.append(val.s[0])
                        break
                else:
                    raise ValueError(
                        "obn {} of node {} (type {}) not found".format(
                            obn, node.name, get_op_type(node)
                        )
                    )
        else:
            conf = get_op_conf(node)
            # it cannot cover all legacy op but it's enough
            if hasattr(conf, "out"):
                out = getattr(conf, "out")
                if isinstance(out, str):
                    outputs = [out]
                else:
                    outputs = out
            else:
                outputs = []
            outputs = ["{}/{}".format(node.name, output) for output in outputs]
        return outputs

    # minimal conversion of attributes
    for node in graph.net.op:
        attr = {}

        op_cnt[get_op_type(node)] += 1

        attrs = node.user_conf.attr.keys() if is_user_op(node) else []
        for a in attrs:
            attr_cnt[a] += 1
            if a == "dtype":
                attr[a] = util.map_flow_dtype(util.get_flow_node_attr(node, "dtype"))
            else:
                attr[a] = util.get_flow_node_attr(node, a)

        try:
            op_type = get_op_type(node)
            input_names = get_inputs(node)
            output_names = get_outputs(node)
            onnx_node = helper.make_node(
                op_type, input_names, output_names, name=node.name, **attr
            )
            onnx_nodes.append(onnx_node)
        except Exception as ex:
            logger.error("pass1 convert failed for %s, ex=%s", node, ex)
            raise

    return onnx_nodes, op_cnt, attr_cnt, dtypes, shape_override


def oneflow_onnx_mapping(g, ops_mapping):
    logger.debug("Mapping Oneflow node to ONNX node(s)")
    mapped_op = collections.Counter()
    unmapped_op = collections.Counter()
    exceptions = []

    ops = list(g.get_nodes())
    for node in ops:
        logger.debug("Process node: %s\n%s", node.name, node.summary)

        if node.need_skip():
            logger.debug("explicitly skip node " + node.name)
            continue

        op = node.type
        map_info = ops_mapping.get(op)
        if map_info is None:
            unmapped_op[op] += 1
            logger.error("oneflow op [%s: %s] is not supported", node.name, op)
            continue
        mapped_op[op] += 1

        func, onnx_op, kwargs = map_info
        if onnx_op is not None:
            node.type = onnx_op
        try:
            func(g, node, **kwargs)
            node.skip_conversion = True
        except Exception as ex:
            logger.error(
                "Failed to convert node %s\n%s", node.name, node.summary, exc_info=1
            )
            exceptions.append(ex)

    return mapped_op, unmapped_op, exceptions


def transpose_inputs(ctx, inputs_as_nchw):
    """Insert a transpose from NHWC to NCHW on model input on users request."""
    ops = []
    for node in ctx.get_nodes():
        for idx, output_name in enumerate(node.output):
            if output_name in inputs_as_nchw:
                shape = ctx.get_shape(output_name)
                if len(shape) != len(constants.NCHW_TO_NHWC):
                    logger.warning(
                        "transpose_input for %s: shape must be rank 4, ignored"
                        % output_name
                    )
                    ops.append(node)
                    continue
                # insert transpose
                op_name = util.make_name(node.name)
                transpose = ctx.insert_new_node_on_output(
                    "Transpose", output_name, name=op_name
                )
                transpose.set_attr("perm", constants.NCHW_TO_NHWC)
                ctx.copy_shape(output_name, transpose.output[0])
                ctx.set_shape(output_name, np.array(shape)[constants.NHWC_TO_NCHW])
                ops.append(transpose)
                ops.append(node)
                continue
        ops.append(node)
    ctx.reset_nodes(ops)


def topological_sort(g, continue_on_error):
    ops = g.get_nodes()
    if not continue_on_error:
        g.topological_sort(ops)
    else:
        try:
            g.topological_sort(ops)
        except:  # pylint: disable=bare-except
            # if we continue on error, ignore graph cycles so we can report all missing ops
            pass


@oneflow_export("onnx.export")
def export(
    job_obj,
    model_save_dir,
    continue_on_error=False,
    target=None,
    opset=None,
    extra_opset=None,
    shape_override=None,
    inputs_as_nchw=None,
):
    assert os.getenv("ENABLE_USER_OP") == "True"
    job_set = c_api_util.GetJobSet()
    job_name = job_obj.__name__
    for job in job_set.job:
        if job.job_conf.job_name == job_name:
            onnx_graph = process_flow_graph(
                job,
                model_save_dir,
                continue_on_error=continue_on_error,
                opset=opset,
                extra_opset=extra_opset,
                shape_override=shape_override,
                inputs_as_nchw=inputs_as_nchw,
            )
            onnx_graph = optimizer.optimize_graph(onnx_graph)
            model_proto = onnx_graph.make_model(job_name)
            return model_proto
    return None


def process_flow_graph(
    flow_graph,
    model_save_dir,
    continue_on_error=False,
    opset=None,
    extra_opset=None,
    shape_override=None,
    inputs_as_nchw=None,
):
    """Convert oneflow graph to onnx graph.
        Args:
            flow_graph: oneflow graph
            continue_on_error: if an op can't be processed (aka there is no mapping), continue
            opset: the opset to be used (int, default is 8)
            extra_opset: list of extra opset's, for example the opset's used by custom ops
            shape_override: dict with inputs that override the shapes given by oneflow
            inputs_as_nchw: transpose inputs in list from nchw to nchw
        Return:
            the onnx model_proto object
    """

    opset = util.find_opset(opset)
    logger.info("Using opset <onnx, %s>", opset)
    if opset > schemas.get_max_supported_opset_version():
        logger.warning(
            "Currently installed onnx package %s is too low to support opset %s, "
            "please upgrade onnx package to avoid potential conversion issue.",
            util.get_onnx_version(),
            opset,
        )

    if shape_override is None:
        shape_override = {}
    if inputs_as_nchw is None:
        inputs_as_nchw = []
    target = constants.DEFAULT_TARGET

    (onnx_nodes, op_cnt, attr_cnt, dtypes, output_shapes,) = oneflow_to_onnx_naive(
        flow_graph, shape_override
    )

    g = Graph(
        onnx_nodes, model_save_dir, output_shapes, dtypes, target, opset, extra_opset,
    )

    # create ops mapping for the desired opsets
    ops_mapping = handler.flow_op.create_mapping(g.opset, g.extra_opset)

    if inputs_as_nchw:
        transpose_inputs(g, inputs_as_nchw)

    # some nodes may already copied into inner Graph, so remove them from main Graph.
    topological_sort(g, continue_on_error)

    mapped_op, unmapped_op, exceptions = oneflow_onnx_mapping(g, ops_mapping)
    if unmapped_op:
        logger.error("Unsupported ops: %s", unmapped_op)
    if exceptions and not continue_on_error:
        raise exceptions[0]

    # onnx requires topological sorting
    topological_sort(g, continue_on_error)

    g.update_proto()

    logger.debug(
        "Summay Stats:\n"
        "\toneflow ops: {}\n"
        "\toneflow attr: {}\n"
        "\tonnx mapped: {}\n"
        "\tonnx unmapped: {}".format(op_cnt, attr_cnt, mapped_op, unmapped_op)
    )

    return g
