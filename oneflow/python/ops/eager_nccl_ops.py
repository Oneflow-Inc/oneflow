from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow


@oneflow_export("eager_nccl_all_reduce")
def eager_nccl_all_reduce(x, device_set, name=None):
    machine_ids = [p[0] for p in device_set]
    device_ids = [p[1] for p in device_set]
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("EagerNcclAllReduce_")
        )
        .Op("eager_nccl_all_reduce")
        .Input("in", [x])
        .Output("out")
        .SetAttr("device_set_machine_ids", machine_ids, "AttrTypeListInt64")
        .SetAttr("device_set_device_ids", device_ids, "AttrTypeListInt64")
        .Build()
        .RemoteBlobList()[0]
    )
