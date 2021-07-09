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
import oneflow
import oneflow._oneflow_internal
import oneflow.core.job.job_set_pb2 as job_set_util

from google.protobuf import text_format


class MultiClientSession(object):
    def __init__(self, sess_id):
        self.is_inited_ = False
        self.is_closed_ = False
        self.sess_ = oneflow._oneflow_internal.RegsiterSession(sess_id)
        oneflow._oneflow_internal.CreateMultiClientSessionContext()
        self.config_proto_ = self._make_config_proto()

    def TryInit(self):
        if not self.is_inited_:
            config_proto_str = text_format.MessageToString(self.config_proto)
            oneflow._oneflow_internal.InitMultiClientSessionContext(config_proto_str)
            self.is_inited_ = True

    def TryClose(self):
        if not self.is_closed_:
            oneflow._oneflow_internal.DestroyMultiClientSessionContext()
            self.is_closed_ = True

    def __def__(self):
        self.TryClose()

    @property
    def id(self):
        return self.sess_.id

    @property
    def config_proto(self):
        if self.config_proto_ is None:
            self.config_proto_ = job_set_util.ConfigProto()
        return self.config_proto_

    def _make_config_proto(self):
        config_proto = job_set_util.ConfigProto()
        config_proto.session_id = self.id
        return config_proto
