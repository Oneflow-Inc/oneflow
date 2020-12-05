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
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.attr_util as attr_util
from oneflow.python.oneflow_export import oneflow_export


class FunctionAttribute(object):
    def __init__(self):
        self.default_placement_scope = None
        self.default_distribute_strategy = None
        self.allow_cpu_return_op = True


class FunctionDesc(object):
    def __init__(self, job_func=None, job_config_proto=None, function_attribute=None):
        if job_config_proto is None:
            job_config_proto = job_conf_pb.JobConfigProto()
        if function_attribute is None:
            function_attribute = FunctionAttribute()
        self.job_func = job_func
        self.job_config_proto = job_config_proto
        self.job_config_proto.predict_conf.SetInParent()
        self.function_attribute = function_attribute
        self.stage_placement_ = None

    def IsTrainable(self):
        if self.job_config_proto.HasField("train_conf"):
            return True
        if self.job_config_proto.HasField("predict_conf"):
            return False
        raise NotImplementedError

    def HasAttr(self, attr_name):
        if attr_name == "flag_name2flag_value":
            return False
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
        elif attr_value.HasField("at_list_int64"):
            return attr_value.at_list_int64
        else:
            raise NotImplementedError()

    def SetAttr(self, attr_name, py_value):
        name2default = session_ctx.GetDefaultSession().function_flag_name2default_val
        assert attr_name in name2default
        flag_name2flag_value = self.job_config_proto.flag_name2flag_value
        default_val = name2default[attr_name]
        attr_util.SetAttrValue(flag_name2flag_value[attr_name], py_value, default_val)

    def SetStagePlacement(
        self, get_stage_partition_scope_ids, stage_partition_strategy
    ):
        self.stage_placement_ = StagePlacement(
            get_stage_partition_scope_ids, stage_partition_strategy
        )

    def ApplyAfterEnvInit(self):
        if self.stage_placement_ is not None:
            self.stage_placement_.Apply(self)


class StagePlacement(object):
    def __init__(self, get_stage_partition_scope_ids, stage_partition_strategy):
        self.get_stage_partition_scope_ids_ = get_stage_partition_scope_ids
        self.stage_partition_strategy_ = stage_partition_strategy

    def Apply(self, function_desc):
        function_desc.SetAttr("enable_stage_partition", True)
        function_desc.SetAttr(
            "stage_partition_scope_ids", self.get_stage_partition_scope_ids_()
        )
        function_desc.SetAttr(
            "stage_partition_strategy", self.stage_partition_strategy_
        )
        function_desc.SetAttr("enable_ssp_variable_proxy", True)
        function_desc.SetAttr("enable_stage_buffer", True)


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
