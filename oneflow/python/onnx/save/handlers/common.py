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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import logging

from oneflow.python.onnx import constants


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring


class BroadcastOp:
    @classmethod
    def Version_6(cls, ctx, node, **kwargs):
        """Elementwise Ops with broadcast flag."""
        shape0 = ctx.get_shape(node.input_tensor_names[0])
        shape1 = ctx.get_shape(node.input_tensor_names[1])
        if shape0 != shape1:
            if (
                shape0
                and shape1
                and len(shape0) < len(shape1)
                and node.op_type in ["Mul", "Add"]
            ):
                tmp = node.input_tensor_names[0]
                node.input_tensor_names[0] = node.input_tensor_names[1]
                node.input_tensor_names[1] = tmp
