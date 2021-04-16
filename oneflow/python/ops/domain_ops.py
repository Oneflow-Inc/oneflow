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

import typing
import oneflow as flow
import oneflow_api
import oneflow.python.framework.id_util as id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("nn.fused_self_attention_query_mul_key_and_value")
def api_fused_self_attention_query_mul_key_and_value(
    x: oneflow_api.BlobDesc,
    head_size: int,
    alpha: float = 1.0,
    name: typing.Optional[str] = None,
) -> oneflow_api.BlobDesc:

    if name is None:
        name = id_util.UniqueStr("FusedSelfAttentionQueryMulKeyAndValue_")

    op = (
        flow.user_op_builder(name)
        .Op("fused_self_attention_query_mul_key_and_value")
        .Input("hidden_states", [x])
        .Attr("head_size", int(head_size))
        .Attr("alpha", float(alpha))
        .Output("query_mul_key")
        .Output("value")
        .Build()
    )

    qmk, v = op.InferAndTryRun().RemoteBlobList()
    return qmk, v
