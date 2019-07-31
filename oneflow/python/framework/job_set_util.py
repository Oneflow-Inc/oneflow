from __future__ import absolute_import

from oneflow.core.job.job_set_pb2 import JobSet
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.compiler as compiler 

@oneflow_export('get_default_job_set')
def get_default_job_set():
    return _default_job_set

@oneflow_export('reset_default_job_set')
def reset_default_job_set():
    global _default_job_set
    _default_job_set = JobSet()
    _job_set2job_name2job_func[id(_default_job_set)] = {}

@oneflow_export('add_job')
def add_job(job_func, job_set = None):
    if job_set == None: job_set = _default_job_set
    compiler.Compile(job_set, job_func)
    _job_set2job_name2job_func[id(job_set)][job_func.__name__] = job_func

def GetJobName2JobFunc(job_set):
    return _job_set2job_name2job_func[id(job_set)]

_job_set2job_name2job_func = {}
_default_job_set = JobSet()
_job_set2job_name2job_func[id(_default_job_set)] = {}
