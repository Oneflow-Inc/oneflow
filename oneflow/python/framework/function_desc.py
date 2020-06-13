from __future__ import absolute_import

import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.oneflow_export import oneflow_export


class FunctionAttribute(object):
    def __init__(self):
        self.default_placement_scope = None
        self.default_distribute_strategy = None
        self.allow_cpu_return_op = True


class FunctionDesc(object):
    def __init__(self, job_func=None, job_config_proto=None, function_attribute=None):
        if job_config_proto is None:
            job_config_proto = job_util.JobConfigProto()
        if function_attribute is None:
            function_attribute = FunctionAttribute()
        self.job_func = job_func
        self.job_config_proto = job_config_proto
        self.job_config_proto.predict_conf.SetInParent()
        self.function_attribute = function_attribute

    def IsTrainable(self):
        if self.job_config_proto.HasField("train_conf"):
            return True
        if self.job_config_proto.HasField("predict_conf"):
            return False
        raise NotImplementedError

    def HasAttr(self, attr_name):
        if attr_name == "flag_name2flag_value":
            return False
        name2default = session_ctx.GetDefaultSession().function_flag_name2default_val
        if attr_name in self.job_config_proto.flag_name2flag_value:
            return True
        return self.job_config_proto.HasField(attr_name)

    def __getattr__(self, attr_name):
        assert attr_name != "flag_name2flag_value"
        flag_name2flag_value = self.job_config_proto.flag_name2flag_value
        name2default = session_ctx.GetDefaultSession().function_flag_name2default_val
        if attr_name not in name2default:
            assert self.job_config_proto.HasField(attr_name)
            return getattr(self.job_config_proto, attr_name)
        attr_value = name2default[attr_name]
        if attr_name in flag_name2flag_value:
            attr_value = flag_name2flag_value[attr_name]
        if attr_value.HasField("at_bool"):
            return attr_value.at_bool
        elif attr_value.HasField("at_int64"):
            return attr_value.at_int64
        elif attr_value.HasField("at_double"):
            return attr_value.at_double
        elif attr_value.HasField("at_string"):
            return attr_value.at_string
        else:
            raise NotImplementedError()


@oneflow_export("current_global_function_desc")
def GetCurrentGlobalFunctionDesc():
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    return session_ctx.GetDefaultSession().GetFunctionDesc(job_name)
