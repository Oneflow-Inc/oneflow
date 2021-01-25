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
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.lib.core.async_util as async_util
import oneflow.python.framework.input_blob_def as input_blob_def_util
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.push_util as push_util
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.eager.op_executor as op_executor
import oneflow_api.oneflow.core.register.logical_blob_id as lbi_util
import oneflow_api

blob_register = blob_register_util.GetDefaultBlobRegister()


def _GetInterfaceBlobObject(builder, op_name):
    if oneflow_api.EagerExecutionEnabled():
        return session_ctx.GetDefaultSession().var_name2var_blob[op_name].blob_object
    blob_object = builder.MakeLazyRefBlobObject(op_name)
    return blob_object


def GetEagerInterfaceBlob(op_name):
    flow.sync_default_session()

    sess = session_ctx.GetDefaultSession()

    def CreateBlob():
        job_name = sess.JobName4InterfaceOpName(op_name)

        def Build(builder, Yield):
            blob_object = _GetInterfaceBlobObject(builder, op_name)
            lbi = lbi_util.LogicalBlobId()
            lbi.set_op_name(op_name)
            op_attribute = sess.OpAttribute4InterfaceOpName(op_name)
            assert len(op_attribute.output_bns) == 1
            lbi.set_blob_name(op_attribute.output_bns[0])
            if blob_object.op_arg_parallel_attr.is_mirrored():
                remote_blob = oneflow_api.EagerMirroredBlob(
                    lbi, blob_object, blob_register, job_name
                )
            else:
                remote_blob = oneflow_api.EagerConsistentBlob(
                    lbi, blob_object, blob_register, job_name
                )

            Yield(remote_blob)

        def AsyncGetInterfaceBlob(Yield):
            vm_util.LogicalRun(lambda builder: Build(builder, Yield))

        blob = async_util.Await(1, AsyncGetInterfaceBlob)[0]
        return blob

    return sess.FindOrCreateLazyBlob(op_name, CreateBlob)


@oneflow_export("experimental.get_interface_blob_value")
def GetInterfaceBlobValue(op_name):
    flow.sync_default_session()

    sess = session_ctx.GetDefaultSession()
    job_name = sess.JobName4InterfaceOpName(op_name)

    def AsyncGetInterfaceBlobValue(Yield):
        def build(builder):
            blob_object = GetEagerInterfaceBlob(op_name).blob_object
            lbi = lbi_util.LogicalBlobId()
            lbi.set_op_name(op_name)
            op_attribute = sess.OpAttribute4InterfaceOpName(op_name)
            assert len(op_attribute.output_bns) == 1
            lbi.set_blob_name(op_attribute.output_bns[0])
            if not isinstance(lbi, lbi_util.LogicalBlobId):
                cfg_lbi = lbi_util.LogicalBlobId()
                cfg_lbi.set_op_name(lbi.op_name)
                cfg_lbi.set_blob_name(lbi.blob_name)
                lbi = cfg_lbi
            if blob_object.op_arg_parallel_attr.is_mirrored():
                remote_blob = oneflow_api.EagerMirroredBlob(
                    lbi, blob_object, blob_register, job_name
                )
            else:
                remote_blob = oneflow_api.EagerConsistentBlob(
                    lbi, blob_object, blob_register, job_name
                )
            if blob_object.op_arg_blob_attr.is_tensor_list:
                value = remote_blob.numpy_list()
            else:
                value = remote_blob.numpy()
            Yield(value)

        vm_util.LogicalRun(build)

    return async_util.Await(1, AsyncGetInterfaceBlobValue)[0]


def FeedValueToInterfaceBlobObject(blob_object, ndarray):
    flow.sync_default_session()

    def build(builder):
        if blob_object.op_arg_blob_attr.is_tensor_list:
            input_blob_def = input_blob_def_util.MirroredTensorListDef(
                [x.shape for x in ndarray],
                dtype=dtype_util.convert_numpy_dtype_to_oneflow_dtype(ndarray.dtype),
            )
        elif blob_object.op_arg_parallel_attr.is_mirrored():
            input_blob_def = input_blob_def_util.MirroredTensorDef(
                ndarray.shape,
                dtype=dtype_util.convert_numpy_dtype_to_oneflow_dtype(ndarray.dtype),
            )
        else:
            input_blob_def = input_blob_def_util.FixedTensorDef(
                ndarray.shape,
                dtype=dtype_util.convert_numpy_dtype_to_oneflow_dtype(ndarray.dtype),
            )
        push_util.FeedValueToEagerBlob(blob_object, input_blob_def, ndarray)

    vm_util.LogicalRun(build)


@oneflow_export("experimental.set_interface_blob_value")
def FeedValueToInterfaceBlob(op_name, ndarray):
    flow.sync_default_session()

    def AsyncFeedValueToInterfaceBlob(Yield):
        def build(builder):
            blob_object = GetEagerInterfaceBlob(op_name).blob_object
            FeedValueToInterfaceBlobObject(blob_object, ndarray)
            Yield()

        vm_util.LogicalRun(build)

    async_util.Await(1, AsyncFeedValueToInterfaceBlob)
