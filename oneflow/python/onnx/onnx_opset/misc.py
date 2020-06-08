# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
misc
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from oneflow.python.onnx.handler import flow_op


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring

@flow_op(["CheckNumerics", "StopGradient"])
class MoveToIdent:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.type = "Identity"
        if node.inputs[0].is_const():
            # should not remove the identity node if it is output of the graph
            if node.output[0] in ctx.outputs:
                return
            # if identity has a const as input, remove it
            input_name = node.input[0]
            output_name = node.output[0]
            ctx.replace_all_inputs(ctx.get_nodes(), output_name, input_name)
            ctx.remove_node(node.name)


@flow_op(['input', 'variable'])
class DirectOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass


@flow_op(["distribute_split"])
class BoxingOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.type = "Identity"


@flow_op("NoOp")
class NukeNode:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        ctx.remove_node(node.name)
