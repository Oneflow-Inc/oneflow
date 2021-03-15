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

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api
from typing import Optional, Union, Sequence


@oneflow_export("count_not_finite")
def count_not_finite(
    x: oneflow_api.BlobDesc, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("CountNotFinite_")
        )
        .Op("count_not_finite")
        .Input("x", [x])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("multi_count_not_finite")
def multi_count_not_finite(
    x: Optional[Sequence[oneflow_api.BlobDesc]] = None, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:

    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("MultiCountNotFinite_")
        )
        .Op("multi_count_not_finite")
        .Input("x", x)
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
