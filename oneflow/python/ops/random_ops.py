from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.id_util as id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("random.bernoulli")
def Bernoulli(x, seed=None, dtype=None, name=None):
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
