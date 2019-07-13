from __future__ import absolute_import

import traceback
import sys
import oneflow.python.framework.ofblob as ofblob
import oneflow_internal

def MakeUserJobInstance(job_name, finish_cb = None):
    if finish_cb is None: finish_cb = _DoNothing
    return JobInstance(job_name, finish_cb = finish_cb)

def MakePullJobInstance(job_name, op_name, pull_cb, finish_cb = None):
    if finish_cb is None: finish_cb = _DoNothing
    return JobInstance(job_name,
                       sole_output_op_name_in_user_job = op_name,
                       pull_cb = pull_cb,
                       finish_cb = finish_cb);

def MakePushJobInstance(job_name, op_name, push_cb, finish_cb = None):
    if finish_cb is None: finish_cb = _DoNothing
    return JobInstance(job_name,
                       sole_input_op_name_in_user_job = op_name,
                       push_cb = push_cb,
                       finish_cb = finish_cb);

def MakeArgPassJobInstance(job_name, src_op_name, dst_op_name, finish_cb = None):
    if finish_cb is None: finish_cb = _DoNothing
    return JobInstance(job_name,
                       sole_output_op_name_in_user_job = src_op_name,
                       sole_input_op_name_in_user_job = dst_op_name,
                       finish_cb = finish_cb);

class JobInstance(oneflow_internal.ForeignJobInstance):
    def __init__(self, job_name,
                 sole_input_op_name_in_user_job = None,
                 sole_output_op_name_in_user_job = None,
                 push_cb = None,
                 pull_cb = None,
                 finish_cb = None):
        oneflow_internal.ForeignJobInstance.__init__(self)
        global _job_name
        global _sole_input_op_name_in_user_job
        global _sole_output_op_name_in_user_job
        global _push_cb
        global _pull_cb
        global _finish_cb
        if job_name: _job_name[id(self)] = str(job_name)
        if sole_input_op_name_in_user_job:
           _sole_input_op_name_in_user_job[id(self)] = str(sole_input_op_name_in_user_job)
        if sole_output_op_name_in_user_job:
           _sole_output_op_name_in_user_job[id(self)] = str(sole_output_op_name_in_user_job)
        if push_cb: _push_cb[id(self)] = push_cb
        if pull_cb: _pull_cb[id(self)] = pull_cb
        if finish_cb: _finish_cb[id(self)] = finish_cb

    def job_name(self): 
        try:
            global _job_name
            ret = _job_name[id(self)]
            del _job_name[id(self)]
            return ret
        except Exception as e:
            print (traceback.format_exc())
            raise e

    def sole_input_op_name_in_user_job(self): 
        try:
            global _sole_input_op_name_in_user_job
            ret = _sole_input_op_name_in_user_job[id(self)]
            del _sole_input_op_name_in_user_job[id(self)]
            return ret
        except Exception as e:
            print (traceback.format_exc())
            raise e
    
    def sole_output_op_name_in_user_job(self): 
        try:
            global _sole_output_op_name_in_user_job
            ret = _sole_output_op_name_in_user_job[id(self)]
            del _sole_output_op_name_in_user_job[id(self)]
            return ret
        except Exception as e:
            print (traceback.format_exc())
            raise e

    def PushBlob(self, of_blob_ptr):
        try:
            global _push_cb
            push_cb = _push_cb[id(self)]
            del _push_cb[id(self)]
            push_cb(ofblob.OfBlob(of_blob_ptr))
        except Exception as e:
            print (traceback.format_exc())
            raise e

    def PullBlob(self, of_blob_ptr):
        try:
            global _pull_cb
            pull_cb = _pull_cb[id(self)]
            del _pull_cb[id(self)]
            pull_cb(ofblob.OfBlob(of_blob_ptr))
        except Exception as e:
            print (traceback.format_exc())
            raise e

    def Finish(self):
        try:
            global _finish_cb
            finish_cb = _finish_cb[id(self)]
            del _finish_cb[id(self)]
            finish_cb()
        except Exception as e:
            print (traceback.format_exc())
            raise e

_job_name = {}
_sole_input_op_name_in_user_job = {}
_sole_output_op_name_in_user_job = {}
_push_cb = {}
_pull_cb = {}
_finish_cb = {}

def _DoNothing():
    pass
