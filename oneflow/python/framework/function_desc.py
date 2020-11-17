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

import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.session_context as session_ctx
import oneflow_api.oneflow.core.job.job_conf as job_conf_cfg
from oneflow.python.oneflow_export import oneflow_export


class FunctionAttribute(object):
    def __init__(self):
        self.default_placement_scope = None
        self.default_distribute_strategy = None
        self.allow_cpu_return_op = True


class FunctionDesc(object):
    def __init__(self, job_func=None, job_config_proto=None, function_attribute=None):
        if job_config_proto is None:
            job_config_proto = job_conf_cfg.JobConfigProto()
        if function_attribute is None:
            function_attribute = FunctionAttribute()
        self.job_func = job_func
        self.job_config_proto = job_config_proto
        self.job_config_proto.mutable_predict_conf()
        self.function_attribute = function_attribute

    def IsTrainable(self):
        if self.job_config_proto.has_train_conf():
            return True
        if self.job_config_proto.has_predict_conf():
            return False
        raise NotImplementedError

    def HasAttr(self, attr_name):
        if attr_name == "flag_name2flag_value":
            return False
        name2default = session_ctx.GetDefaultSession().function_flag_name2default_val
        if attr_name in self.job_config_proto.mutable_flag_name2flag_value():
            return True
        return getattr(self.job_config_proto, "has_" + attr_name)()

    def __getattr__(self, attr_name):
        assert attr_name != "flag_name2flag_value"
        flag_name2flag_value = self.job_config_proto.mutable_flag_name2flag_value()
        name2default = session_ctx.GetDefaultSession().function_flag_name2default_val
        if attr_name not in name2default:
            assert getattr(self.job_config_proto, "has_" + attr_name)()
            return getattr(self.job_config_proto, attr_name)()
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


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def GetCurrentEagerGlobalFunctionDesc():
    sess = session_ctx.GetDefaultSession()
    ret = sess.CurrentEagerGlobalFunctionDesc()
    assert ret is not None
    return ret


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def GetCurrentLazyGlobalFunctionDesc():
    sess = session_ctx.GetDefaultSession()
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    ret = sess.GetLazyFunctionDesc(job_name)
    assert ret is not None
    return ret


@oneflow_export("current_global_function_desc")
def api_current_global_function_desc() -> FunctionDesc:
    api_func = enable_if.unique(
        [GetCurrentLazyGlobalFunctionDesc, GetCurrentEagerGlobalFunctionDesc]
    )
    return api_func()
