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

import os
from typing import Union, Optional, Sequence

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.module as module_util
import oneflow.python.ops.math_unary_elementwise_ops as math_unary_elementwise_ops
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api


@oneflow_export("combined_margin_loss")
def combined_margin_loss(
    x: oneflow_api.BlobDesc,
    label: oneflow_api.BlobDesc,
    m1: float = 1,
    m2: float = 0,
    m3: float = 0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    depth = x.shape[1]
    y, theta = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("CombinedMarginLoss_")
        )
        .Op("combined_margin_loss")
        .Input("x", [x])
        .Input("label", [label])
        .Output("y")
        .Output("theta")
        .Attr("m1", float(m1))
        .Attr("m2", float(m2))
        .Attr("m3", float(m3))
        .Attr("depth", int(depth))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return y
