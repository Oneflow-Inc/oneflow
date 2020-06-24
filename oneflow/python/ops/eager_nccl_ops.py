from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow


@oneflow_export("eager_nccl_all_reduce")
def eager_nccl_all_reduce(x, parallel_conf, name=None):
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("EagerNcclAllReduce_")
        )
        .Op("eager_nccl_all_reduce")
        .Input("in", [x])
        .Output("out")
        .SetAttr("parallel_conf", parallel_conf, "AttrTypeString")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
