from __future__ import absolute_import

import oneflow_internal
from google.protobuf import text_format
from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
from oneflow.core.job.job_set_pb2 import JobSet

def NaiveSequentialRunJobSet(job_set):
    assert(type(job_set) is JobSet)
    oneflow_internal.NaiveSequentialRunJobSet(job_set.SerializeToString())

def InitGlobalOneflowByJobSet(job_set):
    assert(type(job_set) is JobSet)
    oneflow_internal.InitGlobalOneflowBySerializedJobSet(job_set.SerializeToString())

def GetInterUserJobInfo():
    return text_format.Parse(oneflow_internal.GetSerializedInterUserJobInfo(), InterUserJobInfo())

def LaunchJob(job_instance):
    oneflow_internal.LaunchJob(job_instance)

def DestroyGlobalOneflow():
    oneflow_internal.DestroyGlobalOneflow()
