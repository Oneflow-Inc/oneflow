# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
misc
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import logging

from oneflow.python.onnx.handler import flow_op


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring


@flow_op(["input", "return", "variable"], None)
class DirectOp:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        pass


@flow_op(
    ["distribute_split", "distribute_concat", "distribute_clone", "distribute_add"],
    "Identity",
)
class BoxingOp:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        pass
