from __future__ import absolute_import

inter_user_job_info = None

def InitInterUserJobInfo(info):
    global inter_user_job_info
    inter_user_job_info = info
    
def DestroyInterUserJobInfo():
    global inter_user_job_info
    inter_user_job_info = None

job_instance_pre_launch_callbacks = []

def AddJobInstancePreLaunchCallbacks(cb):
    global job_instance_pre_launch_callbacks
    job_instance_pre_launch_callbacks.append(cb)

job_instance_post_finish_callbacks = []

def AddJobInstancePostFinishCallbacks(cb):
    global job_instance_post_finish_callbacks
    job_instance_post_finish_callbacks.append(cb)
