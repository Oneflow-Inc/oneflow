# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
controlflow
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oneflow.python.onnx import utils
from oneflow.python.onnx.handler import flow_op


# pylint: disable=unused-argument,missing-docstring

@flow_op("Where", onnx_op='NonZero')
class Where:
    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # T_y output = Where(T_x condition), return indices of elements whose value are True
        # in onnx, indices are returned in this way [[ind_a_0, ind_b_0, ...], [ind_a_1, ind_b_1,...]];
        # while in tf, the result will be [[ind_a_0, ind_a_1, ...], [ind_b_0, ind_b_1, ...], ...]
        # this is the reason a transpose node inserted here.
        transpose_node = ctx.insert_new_node_on_output("Transpose",
                                                       node.output[0], name=utils.make_name("where_op_added"))
        ctx.copy_shape(node.output[0], transpose_node.output[0])
        ctx.copy_dtype(node.output[0], transpose_node.output[0])
