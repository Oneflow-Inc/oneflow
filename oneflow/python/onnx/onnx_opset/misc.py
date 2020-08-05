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
