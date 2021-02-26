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
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# oneflow.python.onnx.oneflow.python.onnx - rewrite oneflow graph to onnx graph

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import itertools
import logging
import os
import os.path
import sys
import traceback
from typing import Text, Optional, Dict, Callable, List

import numpy as np
from onnx import helper, onnx_pb

import oneflow
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.onnx
from oneflow.python.onnx import constants, schemas, util
from oneflow.python.onnx.save import handler, optimizer
from oneflow.python.onnx.onnx_wrapper import Graph
import oneflow.python.onnx.save.handlers  # pylint: disable=unused-import

logger = logging.getLogger(__name__)


def FlowToOnnxNaive(graph, shape_override):
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
        dtypes[lbn] = util.Flow2OnnxDtype(lbd.body.data_type)

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
                attr[a] = util.Flow2OnnxDtype(util.get_flow_node_attr(node, "dtype"))
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


def FlowOnnxMapping(g, ops_mapping):
    logger.debug("Mapping Oneflow node to ONNX node(s)")
    mapped_op = collections.Counter()
    unmapped_op = collections.Counter()
    exceptions = []

    ops = list(g.get_nodes())
    for node in ops:
        logger.debug("Process node: %s\n%s", node.name, node.summary)

        if node.skip_conversion:
            logger.debug("explicitly skip node " + node.name)
            continue

        op = node.op_type
        map_info = ops_mapping.get(op)
        if map_info is None:
            unmapped_op[op] += 1
            logger.error("oneflow op [%s: %s] is not supported", node.name, op)
            continue
        mapped_op[op] += 1

        func, onnx_op, kwargs = map_info
        if onnx_op is not None:
            node.op_type = onnx_op
        try:
            func(g, node, **kwargs)
            node.skip_conversion = True
        except Exception as ex:
            logger.error(
                "Failed to convert node %s\n%s", node.name, node.summary, exc_info=1
            )
            exceptions.append(ex)

    return mapped_op, unmapped_op, exceptions


def TopologicalSort(g, continue_on_error):
    ops = g.get_nodes()
    if not continue_on_error:
        g.TopologicalSort(ops)
    else:
        try:
            g.TopologicalSort(ops)
        except:  # pylint: disable=bare-except
            # if we continue on error, ignore graph cycles so we can report all missing ops
            pass


@session_ctx.try_init_default_session
@oneflow_export("onnx.export")
def Export(
    job_func: Callable,
    model_save_dir: Text,
    onnx_filename: Text,
    continue_on_error: bool = False,
    opset: Optional[int] = None,
    extra_opset: Optional[int] = None,
    shape_override: Optional[Dict[Text, List[int]]] = None,
    external_data: bool = False,
):
    r"""Export a oneflow model into ONNX format.

    Args:
        job_func: The job function
        model_save_dir: The directory containing oneflow model weights. Users are expected to call check_point.save(dir), wait for the model saving finishing, and pass the argument 'dir' as model_save_dir.
        onnx_filename: a string for the output filename
        continue_on_error: if an op can't be processed (aka there is no mapping), continue
        opset: the opset to be used (int, default is oneflow.python.onnx.constants.PREFERRED_OPSET)
        extra_opset: list of extra opset's, for example the opset's used by custom ops
        shape_override: dict with inputs that override the shapes given by oneflow
        external_data: Save weights as ONNX external data, usually to bypass the 2GB file size limit of protobuf.
    """
    assert os.getenv("ENABLE_USER_OP") != "False"
    assert os.path.isdir(model_save_dir)
    job_set = c_api_util.GetJobSet()
    job_name = job_func.__name__
    for job in job_set.job:
        # TODO(OYY) Modify the interface before modifying it
        if job.job_conf.job_name == job_name:
            onnx_graph = ProcessFlowGraph(
                job,
                model_save_dir,
                continue_on_error=continue_on_error,
                opset=opset,
                extra_opset=extra_opset,
                shape_override=shape_override,
            )
            onnx_graph = optimizer.OptimizeGraph(onnx_graph)
            model_proto = onnx_graph.MakeModel(
                job_name, onnx_filename, external_data=external_data
            )
            with open(onnx_filename, "wb") as f:
                try:
                    f.write(model_proto.SerializeToString())
                except ValueError as e:
                    raise ValueError(
                        "Error occured when running model_proto.SerializeToString(). If the model is larger than 2GB, please specify external_data=True when calling flow.onnx.export. Original error message:\n{}".format(
                            e
                        )
                    )
            return
    raise ValueError('Cannot find job "{}" in jobset'.format(job_name))


def ProcessFlowGraph(
    flow_graph,
    model_save_dir,
    continue_on_error=False,
    opset=None,
    extra_opset=None,
    shape_override=None,
):
    opset = util.FindOpset(opset)
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

    (onnx_nodes, op_cnt, attr_cnt, dtypes, output_shapes,) = FlowToOnnxNaive(
        flow_graph, shape_override
    )

    g = Graph(onnx_nodes, model_save_dir, output_shapes, dtypes, opset, extra_opset,)

    # create ops mapping for the desired opsets
    ops_mapping = handler.flow_op.CreateMapping(g.opset, g.extra_opset)

    # some nodes may already copied into inner Graph, so remove them from main Graph.
    TopologicalSort(g, continue_on_error)

    mapped_op, unmapped_op, exceptions = FlowOnnxMapping(g, ops_mapping)
    if unmapped_op:
        logger.error("Unsupported ops: %s", unmapped_op)
    if exceptions and not continue_on_error:
        raise exceptions[0]

    # onnx requires topological sorting
    TopologicalSort(g, continue_on_error)

    g.UpdateProto()

    logger.debug(
        "Summay Stats:\n"
        "\toneflow ops: {}\n"
        "\toneflow attr: {}\n"
        "\tonnx mapped: {}\n"
        "\tonnx unmapped: {}".format(op_cnt, attr_cnt, mapped_op, unmapped_op)
    )

    return g
