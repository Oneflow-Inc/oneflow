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
