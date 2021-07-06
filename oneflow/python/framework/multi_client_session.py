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
import oneflow.python.framework.env_util as env_util
import oneflow.python.framework.session_context as session_ctx

from oneflow.python.framework.session import Session


class MultiClientSession(Session):
    def __init__(self, sess_id):
        super(self).__init__(sess_id)
        # self.context_ = oneflow._oneflow_internal.CreateMultiClientSessionContext(sess_id)

    def init(self):
        # self.context_.InitGlobalObjects(self.config_proto)
        pass

    def close(self):
        pass
