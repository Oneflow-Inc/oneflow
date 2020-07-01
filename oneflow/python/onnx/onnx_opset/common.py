# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
common
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from oneflow.python.onnx import constants


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring


class BroadcastOp:
    @classmethod
    def Version_6(cls, ctx, node, **kwargs):
        """Elementwise Ops with broadcast flag."""
        shape0 = ctx.get_shape(node.input[0])
        shape1 = ctx.get_shape(node.input[1])
        if shape0 != shape1:
            if (
                shape0
                and shape1
                and len(shape0) < len(shape1)
                and node.type in ["Mul", "Add"]
            ):
                tmp = node.input[0]
                node.input[0] = node.input[1]
                node.input[1] = tmp
