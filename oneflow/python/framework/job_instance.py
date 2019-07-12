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
        job_name = str(job_name)
        sole_input_op_name_in_user_job = str(sole_input_op_name_in_user_job)
        sole_output_op_name_in_user_job = str(sole_output_op_name_in_user_job)
        self.job_name_ = job_name
        self.sole_input_op_name_in_user_job_ = sole_input_op_name_in_user_job
        self.sole_output_op_name_in_user_job_ = sole_output_op_name_in_user_job
        self.push_cb_ = push_cb
        self.pull_cb_ = pull_cb
        self.finish_cb_ = finish_cb

    def job_name(self): return self.job_name_

    def sole_input_op_name_in_user_job(self): return self.sole_input_op_name_in_user_job_
    
    def sole_output_op_name_in_user_job(self): return self.sole_output_op_name_in_user_job_

    def PushBlob(self, of_blob_ptr):
        try:
            self.push_cb_(ofblob.OfBlob(of_blob_ptr))
        except Exception as e:
            print (traceback.format_exc())
            raise e

    def PullBlob(self, of_blob_ptr):
        try:
            self.pull_cb_(ofblob.OfBlob(of_blob_ptr))
        except Exception as e:
            print (traceback.format_exc())
            raise e

    def Finish(self):
        try:
            self.finish_cb_()
        except Exception as e:
            print (traceback.format_exc())
            raise e

def _DoNothing():
    pass
