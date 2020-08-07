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
import oneflow as flow
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.lib.core.async_util as async_util
import oneflow.python.framework.input_blob_def as input_blob_def
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.push_util as push_util
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("experimental.get_interface_blob_value")
def GetInterfaceBlobValue(op_name):
    sess = session_ctx.GetDefaultSession()
    job_name = sess.JobName4InterfaceOpName(op_name)

    def AsyncGetInterfaceBlobValue(Yield):
        def build(builder):
            blob_object = builder.MakeLazyRefBlobObject(op_name)
            lbi = logical_blob_id_util.LogicalBlobId()
            lbi.op_name = op_name
            op_attribute = sess.OpAttribute4InterfaceOpName(op_name)
            assert len(op_attribute.output_bns) == 1
            lbi.blob_name = op_attribute.output_bns[0]
            Yield(
                remote_blob_util.EagerConsistentBlob(lbi, blob_object, job_name).numpy()
            )

        vm_util.LogicalRun(build)

    return async_util.Await(1, AsyncGetInterfaceBlobValue)[0]


@oneflow_export("experimental.set_interface_blob_value")
def FeedValueToInterfaceBlob(op_name, ndarray):
    flow.sync_default_session()

    def AsyncFeedValueToInterfaceBlob(Yield):
        def build(builder):
            blob_object = builder.MakeLazyRefBlobObject(op_name)
            push_util.FeedValueToEagerBlob(
                blob_object,
                input_blob_def.FixedTensorDef(ndarray.shape, dtype=flow.float),
                ndarray,
            )
            Yield()

        vm_util.LogicalRun(build)

    async_util.Await(1, AsyncFeedValueToInterfaceBlob)
