import oneflow.core.job.job_pb2 as job_util

from contextlib import contextmanager

def CurrentJobConf(): return  job_conf_stack[0]

@contextmanager
def JobConfScope(job_conf):
    global job_conf_stack
    job_conf_stack.insert(0, job_conf);
    yield
    job_conf_stack.pop(0)

def GetInitialJobConf(job_name):
    job_conf = job_util.JobConfigProto()
    job_conf.job_name = job_name
    return job_conf

job_conf_stack = [GetInitialJobConf("__InitialJob__")]
