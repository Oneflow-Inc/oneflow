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
from google.protobuf import text_format

import oneflow._oneflow_internal
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.python.framework.c_api_util as c_api_util


class MultiClientSession(object):
    def __init__(self, sess_id):
        self.is_inited_ = False
        self.is_closed_ = False
        self.sess_ = oneflow._oneflow_internal.RegsiterSession(sess_id)
        oneflow._oneflow_internal.CreateMultiClientSessionContext()
        self.config_proto_ = self._make_config_proto()

        self.function_flag_name2default_val_ = {}
        self._update_function_flag_name2defaultVal()

        self.scope_attr_name2default_val_ = {}
        self._update_scope_attr_name2defaultVal()

    def TryInit(self):
        if not self.is_inited_:
            print("self.config_proto:", self.config_proto)
            config_proto_str = text_format.MessageToString(self.config_proto)
            print("str:", config_proto_str)
            oneflow._oneflow_internal.InitMultiClientSessionContext(config_proto_str)
            self.is_inited_ = True

    def TryClose(self):
        if not self.is_closed_:
            oneflow._oneflow_internal.DestroyMultiClientSessionContext()
            oneflow._oneflow_internal.ClearSessionById(self.id)
            self.is_closed_ = True

    def __del__(self):
        self.TryClose()

    @property
    def id(self):
        return self.sess_.id

    @property
    def config_proto(self):
        return self.config_proto_

    @property
    def function_flag_name2default_val(self):
        return self.function_flag_name2default_val_

    @property
    def scope_attr_name2default_val(self):
        return self.scope_attr_name2default_val_

    def _make_config_proto(self):
        config_proto = job_set_util.ConfigProto()
        config_proto.resource.machine_num = oneflow._oneflow_internal.GetNodeSize()
        if oneflow._oneflow_internal.flags.with_cuda():
            config_proto.resource.gpu_device_num = 1
        else:
            config_proto.resource.cpu_device_num = 1
            config_proto.resource.gpu_device_num = 0
        config_proto.session_id = self.id
        return config_proto

    def _update_function_flag_name2defaultVal(self):
        items = c_api_util.GetFunctionConfigDef().attr_name2attr_def.items()
        self.function_flag_name2default_val_ = {k: v.default_val for k, v in items}

    def _update_scope_attr_name2defaultVal(self):
        items = c_api_util.GetScopeConfigDef().attr_name2attr_def.items()
        self.scope_attr_name2default_val_ = {k: v.default_val for k, v in items}

    def AnyGlobalFunctionDefined(self):  # compatible with old version session
        return False

    @property
    def is_running(self):  # compatible with old version session
        return self.is_inited_
