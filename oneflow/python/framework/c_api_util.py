from __future__ import absolute_import

from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
import oneflow.core.job.job_set_pb2 as job_set_util
from google.protobuf import text_format
import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.framework.runtime_context as runtime_ctx

def IsOpTypeCaseCpuSupportOnly(op_type_case):
    return oneflow_internal.IsOpTypeCaseCpuSupportOnly(op_type_case)

def Init(config_proto):
    assert(type(config_proto) is job_set_util.ConfigProto)
    config_proto_str = text_format.MessageToString(config_proto)
    oneflow_internal.InitBySerializedConfigProto(config_proto_str)

def InitGlobalOneflowByJobSet(job_set):
    assert(type(job_set) is job_set_util.JobSet)
    job_set_str = text_format.MessageToString(job_set)
    oneflow_internal.InitGlobalOneflowBySerializedJobSet(job_set_str)

def GetInterUserJobInfo():
    return text_format.Parse(oneflow_internal.GetSerializedInterUserJobInfo(), InterUserJobInfo())

def LaunchJob(job_instance):
    for pre_launch_callback in runtime_ctx.job_instance_pre_launch_callbacks:
        pre_launch_callback(job_instance)
    for post_finish_callback in runtime_ctx.job_instance_post_finish_callbacks:
        job_instance.AddPostFinishCallback(post_finish_callback)
    oneflow_internal.LaunchJob(job_instance)

def DestroyGlobalOneflow():
    oneflow_internal.DestroyGlobalOneflow()

def DestroyGlobalEnvironment():
    oneflow_internal.DestroyGlobalEnvironment()
