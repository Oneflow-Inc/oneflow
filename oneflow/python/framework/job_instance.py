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
from __future__ import absolute_import

import sys
import traceback

import oneflow.python.framework.ofblob as ofblob
import oneflow_api


def MakeUserJobInstance(job_name, finish_cb=None):
    return MakeJobInstance(job_name, finish_cb=finish_cb)


def MakePullJobInstance(job_name, op_name, pull_cb, finish_cb=None):
    return MakeJobInstance(
        job_name,
        sole_output_op_name_in_user_job=op_name,
        pull_cb=pull_cb,
        finish_cb=finish_cb,
    )


def MakePushJobInstance(job_name, op_name, push_cb, finish_cb=None):
    return MakeJobInstance(
        job_name,
        sole_input_op_name_in_user_job=op_name,
        push_cb=push_cb,
        finish_cb=finish_cb,
    )


def MakeArgPassJobInstance(job_name, src_op_name, dst_op_name, finish_cb=None):
    return MakeJobInstance(
        job_name,
        sole_output_op_name_in_user_job=src_op_name,
        sole_input_op_name_in_user_job=dst_op_name,
        finish_cb=finish_cb,
    )


def MakeJobInstance(*arg, **kw):
    def _DoNothing():
        pass

    if "finish_cb" not in kw or kw["finish_cb"] is None:
        kw["finish_cb"] = _DoNothing
    job_instance = JobInstance(*arg, **kw)
    # python object lifetime is a headache
    # _flying_job_instance prevents job_instance earlier destructation
    global _flying_job_instance
    _flying_job_instance[id(job_instance)] = job_instance

    def DereferenceJobInstance(job_instance):
        global _flying_job_instance
        del _flying_job_instance[id(job_instance)]

    job_instance.AddPostFinishCallback(DereferenceJobInstance)
    return job_instance


class JobInstance(oneflow_api.ForeignJobInstance):
    def __init__(
        self,
        job_name,
        sole_input_op_name_in_user_job=None,
        sole_output_op_name_in_user_job=None,
        push_cb=None,
        pull_cb=None,
        finish_cb=None,
    ):
        oneflow_api.ForeignJobInstance.__init__(self)
        self.thisown = 0
        self.job_name_ = str(job_name)
        self.sole_input_op_name_in_user_job_ = str(sole_input_op_name_in_user_job)
        self.sole_output_op_name_in_user_job_ = str(sole_output_op_name_in_user_job)
        self.push_cb_ = push_cb
        self.pull_cb_ = pull_cb
        self.finish_cb_ = finish_cb
        self.post_finish_cbs_ = []

    def job_name(self):
        try:
            return self.job_name_
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def sole_input_op_name_in_user_job(self):
        try:
            return self.sole_input_op_name_in_user_job_
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def sole_output_op_name_in_user_job(self):
        try:
            return self.sole_output_op_name_in_user_job_
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def PushBlob(self, of_blob_ptr):
        try:
            self.push_cb_(ofblob.OfBlob(of_blob_ptr))
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def PullBlob(self, of_blob_ptr):
        try:
            self.pull_cb_(ofblob.OfBlob(of_blob_ptr))
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def Finish(self):
        try:
            self.finish_cb_()
        except Exception as e:
            print(traceback.format_exc())
            raise e
        finally:
            try:
                for post_finish_cb in self.post_finish_cbs_:
                    post_finish_cb(self)
            except Exception as e:
                print(traceback.format_exc())
                raise e

    def AddPostFinishCallback(self, cb):
        self.post_finish_cbs_.append(cb)


# span python object lifetime
_flying_job_instance = {}
