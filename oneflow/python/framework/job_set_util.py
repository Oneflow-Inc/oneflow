from __future__ import absolute_import

from oneflow.core.job.job_set_pb2 import JobSet
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.compiler as compiler

def add_job(job_func):
    assert _job_name2job_func_mutable == True
    if job_func.__name__ not in _job_name2job_func:
        _job_name2job_func[job_func.__name__] = job_func

def compile_all_job(job_set = None):
    assert _job_name2job_func_mutable == True
    _job_name2job_funcs_mutable = False
    if job_set == None: job_set = _default_job_set
    for job_name, job_func in _job_name2job_func.items():
        compiler.Compile(job_set, job_func)
        _job_set2job_name2job_func[id(job_set)][job_name] = job_func

@oneflow_export('get_default_job_set')
def get_default_job_set():
    return _default_job_set

@oneflow_export('reset_default_job_set')
def reset_default_job_set():
    global _default_job_set
    _default_job_set = JobSet()
    _job_set2job_name2job_func[id(_default_job_set)] = {}

@oneflow_export('inter_job_reuse_mem_strategy')
def inter_job_reuse_mem_strategy(strategy_str, job_set = None, **kwargs):
    assert type(strategy_str) is str
    if job_set == None: job_set = _default_job_set
    if strategy_str == "reuse_mem_priority":
        job_set.inter_job_reuse_mem_strategy.reuse_mem_priority.SetInParent()
        assert job_set.inter_job_reuse_mem_strategy.HasField("reuse_mem_priority")
    elif strategy_str == "parallelism_priority":
        job_set.inter_job_reuse_mem_strategy.parallelism_priority.SetInParent()
        assert job_set.inter_job_reuse_mem_strategy.HasField("parallelism_priority")
    elif strategy_str == "custom_parallelism":
        assert kwargs["job_name_groups"] is not None
        for job_name_group in kwargs["job_name_groups"]:
            group = job_set.inter_job_reuse_mem_strategy.custom_parallelism.nonparallel_group.add()
            for job_name in job_name_group:
                assert type(job_name) is str
                group.job_name.append(job_name)

def GetJobName2JobFunc(job_set):
    return _job_set2job_name2job_func[id(job_set)]

_job_name2job_func = {}
_job_name2job_func_mutable = True
_job_set2job_name2job_func = {}
_default_job_set = JobSet()
_job_set2job_name2job_func[id(_default_job_set)] = {}
