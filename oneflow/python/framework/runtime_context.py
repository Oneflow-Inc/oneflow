from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util

inter_user_job_info = None

def Init():
    global inter_user_job_info
    inter_user_job_info = c_api_util.GetInterUserJobInfo()
    
def Destroy():
    global inter_user_job_info
    inter_user_job_info = None
