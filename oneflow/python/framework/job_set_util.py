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

@oneflow_export('job_mem_sharing_strategy')
def job_mem_sharing_strategy(strategy_str, job_set = None, **kwargs):
    assert type(strategy_str) is str
    if job_set == None: job_set = _default_job_set
    if strategy_str == "mem_sharing_priority":
        job_set.job_mem_sharing_strategy.mem_sharing_priority.SetInParent()
        assert job_set.job_mem_sharing_strategy.HasField("mem_sharing_priority")
    elif strategy_str == "parallelism_priority":
        job_set.job_mem_sharing_strategy.parallelism_priority.SetInParent()
        assert job_set.job_mem_sharing_strategy.HasField("parallelism_priority")
    elif strategy_str == "custom_parallelism":
        assert kwargs["job_name_groups"] is not None
        for job_name_group in kwargs["job_name_groups"]:
            group = job_set.job_mem_sharing_strategy.custom_parallelism.nonparallel_group.add()
            for job_name in job_name_group:
                assert type(job_name) is str
                group.job_name.append(job_name)

def GetJobName2JobFunc(job_set):
    return _job_set2job_name2job_func[id(job_set)]

_job_set2job_name2job_func = {}
_default_job_set = JobSet()
_job_set2job_name2job_func[id(_default_job_set)] = {}
