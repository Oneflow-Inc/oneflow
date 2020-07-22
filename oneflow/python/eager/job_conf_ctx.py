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
import oneflow.core.job.job_conf_pb2 as job_conf_pb

from contextlib import contextmanager


def CurrentJobConf():
    return job_conf_stack[0]


@contextmanager
def JobConfScope(job_conf):
    global job_conf_stack
    job_conf_stack.insert(0, job_conf)
    yield
    job_conf_stack.pop(0)


def GetInitialJobConf(job_name):
    job_conf = job_conf_pb.JobConfigProto()
    job_conf.job_name = job_name
    return job_conf


job_conf_stack = [GetInitialJobConf("__InitialJob__")]
