from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_context as runtime_ctx
import oneflow.python.framework.job_instance as job_instance
import threading

def AsyncPush(op_name, push_data_cb):
    push_job_name = runtime_ctx.inter_user_job_info.input_or_var_op_name2push_job_name[op_name]
    c_api_util.LaunchJob(job_instance.MakePushJobInstance(push_job_name, op_name, push_data_cb))

def AsyncPull(op_name, pull_data_cb):
    pull_job_name = runtime_ctx.inter_user_job_info.output_or_var_op_name2pull_job_name[op_name]
    c_api_util.LaunchJob(job_instance.MakePullJobInstance(pull_job_name, op_name, pull_data_cb))
    
def MakePushCallback(ndarray):
    return lambda ofblob: ofblob.CopyFromNdarrayOrNestedNdarrayList(ndarray)

