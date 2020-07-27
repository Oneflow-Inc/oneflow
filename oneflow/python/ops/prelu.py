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
from typing import Optional, Sequence
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.remote_blob as remote_blob_util


@oneflow_export("layers.prelu")
def prelu(
    inputs: remote_blob_util.BlobDef,
    alpha_initializer: Optional[op_conf_util.InitializerConf] = None,
    alpha_regularizer: Optional[op_conf_util.RegularizerConf] = None,
    shared_axes: Optional[Sequence[int]] = None,
    trainable: bool = True,
    name: str = "PRelu",
    model_distribute: distribute_util.Distribute = distribute_util.broadcast(),
) -> remote_blob_util.BlobDef:
    alpha_shape = list(inputs.shape[1:])
    if shared_axes is not None:
        for i in shared_axes:
            assert i >= 1 and i < len(inputs.shape)
            alpha_shape[i - 1] = 1

    if alpha_initializer is None:
        alpha_initializer = flow.constant_initializer(0)

    with flow.scope.namespace(name):
        alpha = flow.get_variable(
            name="alpha",
            shape=alpha_shape,
            dtype=inputs.dtype,
            initializer=alpha_initializer,
            regularizer=alpha_regularizer,
            trainable=trainable,
            distribute=model_distribute,
            reuse=False,
        )

    op = (
        flow.user_op_builder(name)
        .Op("prelu")
        .Input("x", [inputs])
        .Input("alpha", [alpha])
        .Output("y")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()
