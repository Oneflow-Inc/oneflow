from __future__ import absolute_import

from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
from oneflow.core.job.job_set_pb2 import JobSet
from google.protobuf import text_format
import oneflow_internal
import oneflow.python.framework.runtime_context as runtime_ctx

def NaiveSequentialRunJobSet(job_set):
    assert(type(job_set) is JobSet)
    oneflow_internal.NaiveSequentialRunJobSet(job_set.SerializeToString())

def InitGlobalOneflowByJobSet(job_set):
    assert(type(job_set) is JobSet)
    oneflow_internal.InitGlobalOneflowBySerializedJobSet(job_set.SerializeToString())

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
