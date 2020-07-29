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

from typing import Optional, Sequence, Union

import oneflow as flow
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("random.bernoulli")
def Bernoulli(
    x: remote_blob_util.BlobDef,
    seed: Optional[int] = None,
    dtype: Optional[dtype_util.dtype] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if name is None:
        name = id_util.UniqueStr("Bernoulli_")
    if dtype is None:
        dtype = x.dtype

    return (
        flow.user_op_builder(name)
        .Op("bernoulli")
        .Input("in", [x])
        .Output("out")
        .Attr("dtype", dtype)
        .SetRandomSeed(seed)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
