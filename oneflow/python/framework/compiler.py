from __future__ import absolute_import

import oneflow.core.job.job_conf_pb2 as job_conf_util
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.python.framework.decorator_context as decorator_context
import oneflow.python.framework.oneflow_mode as oneflow_mode
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.config_util as config_util


def Compile():
    assert oneflow_mode.IsCurrentCompileMode(), "Compile() must be under compile mode"
    assert decorator_context.main_func is not None, "no main function found"
    assert len(decorator_context.job_name2func) > 0, "no job function found"
    compile_context.cur_job_set = job_set_util.JobSet()
    with compile_context.CompilingMain():
        job_set = compile_context.cur_job_set
        decorator_context.main_func.__oneflow_config_func__(job_set)
        config_util.DefaultConfigJobSet(job_set)
    with compile_context.CompilingRemote():
        for job_name, func in decorator_context.job_name2func.items():
            compile_context.cur_job = job_conf_util.JobConf()
            func.__oneflow_config_func__(compile_context.cur_job)
            config_util.DefaultConfigJobConf(compile_context.cur_job)
            compile_context.cur_job.job_name = func.__name__
            job_set.job_conf.add().CopyFrom(compile_context.cur_job)
    return job_set

def GetMainFunc():
    return decorator_context.main_func

def CompileJob(func):
    func(*arg)
