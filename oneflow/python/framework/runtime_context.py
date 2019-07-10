from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_ctx
import oneflow.python.framework.c_api_util as c_api_util

job_name2input_op_names = {}

job_name2output_op_names = {}

inter_user_job_info = None

def Init():
    global job_name2input_op_names
    job_name2input_op_names = {
        k : v.op_name for k, v in compile_ctx.job_name2input_logical_blobs.items()
    }
    
    global job_name2output_op_names
    job_name2output_op_names = {
        k : v.op_name for k, v in compile_ctx.job_name2output_logical_blobs.items()
    }
    
    global inter_user_job_info
    inter_user_job_info = c_api_util.GetInterUserJobInfo()
    
def Destroy():
    global job_name2input_op_names
    job_name2input_op_names = {}

    global job_name2output_op_names
    job_name2output_op_names = {}

    global inter_user_job_info
    inter_user_job_info = None
