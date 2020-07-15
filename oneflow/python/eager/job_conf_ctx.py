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
