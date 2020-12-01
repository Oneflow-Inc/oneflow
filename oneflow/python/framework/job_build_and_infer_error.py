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

from google.protobuf import text_format
import oneflow.python.framework.session_context as session_ctx

import traceback
import os


class JobBuildAndInferError(Exception):
    def __init__(self, error_proto):
        assert error_proto.HasField("error_type")
        self.error_proto_ = error_proto
        self.error_summary_ = self.error_proto_.error_summary
        self.error_proto_.ClearField("error_summary")
        self.msg_ = self.error_proto_.msg
        self.error_proto_.ClearField("msg")
        resource = session_ctx.GetDefaultSession().config_proto.resource

        def get_op_kernel_not_found_error_str(error_proto):
            error_msg = str(self.error_proto_.op_kernel_not_found_error)
            error_msg = error_msg.replace("\\", "")
            error_msg = error_msg.replace("op_kernels_not_found_debug_str:", "")
            error_msg = "\n".join(
                [e.strip()[1:-1] for e in error_msg.strip().split("\n")]
            )

            return (
                "\n\nFailure messages of registered kernels for current Op node: \n%s"
                % error_msg
            )

        def get_multiple_op_kernels_matched_error_str(error_proto):
            error_msg = str(self.error_proto_.multiple_op_kernels_matched_error)
            error_msg = error_msg.replace("\\", "")
            error_msg = error_msg.replace("matched_op_kernels_debug_str:", "")
            error_msg = "\n".join(
                [e.strip()[1:-1] for e in error_msg.strip().split("\n")]
            )

            return (
                "\n\nThere exists multiple registered kernel candidates for current Op node: \n%s"
                % error_msg
            )

        self.error_type2get_error_str = {
            "op_kernel_not_found_error": get_op_kernel_not_found_error_str,
            "multiple_op_kernels_matched_error": get_multiple_op_kernels_matched_error_str,
        }

    def __str__(self):

        ret = (
            "\n\nerror msg: \n\n\033[1;31m%s\033[0m" % str(self.error_summary_).strip()
        )

        oneof_field = self.error_proto_.WhichOneof("error_type")
        if oneof_field in self.error_type2get_error_str:
            ret += self.error_type2get_error_str[oneof_field](self.error_proto_)
            self.error_proto_.ClearField(oneof_field)
            ret += "\n%s" % text_format.MessageToString(self.error_proto_)
        else:
            ret += "\n%s" % str(self.error_proto_)

        ret += "\n%s" % str(self.msg_).strip()

        return ret
