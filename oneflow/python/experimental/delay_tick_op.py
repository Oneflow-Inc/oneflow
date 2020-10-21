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

from typing import Optional, Tuple
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.dtype as dtype_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("experimental.delay_tick")
def delay_tick(
    x: input_blob_util.ArgBlobDef, delay_num: int = 0, name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr("DelayTick_")
    else:
        op_conf.name = name
    op_conf.delay_tick_conf.tick = x.unique_name
    op_conf.delay_tick_conf.out = "out"
    op_conf.delay_tick_conf.delay_num = delay_num

    interpret_util.Forward(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    out_lbi.op_name = op_conf.name
    out_lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(out_lbi)
