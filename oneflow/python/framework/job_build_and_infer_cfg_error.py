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
import oneflow_api.oneflow.core.common.error as error_cfg

import traceback
import os


class JobBuildAndInferCfgError(Exception):
    def __init__(self, error_cfg):
        assert error_cfg.has_error_type()
        self.error_cfg_ = error_cfg
        self.error_summary_ = self.error_cfg_.error_summary()
        self.error_cfg_.clear_error_summary()
        self.msg_ = self.error_cfg_.msg()
        self.error_cfg_.clear_msg()
        resource = session_ctx.GetDefaultSession().config_proto.resource
        if resource.enable_debug_mode == False:
            self.error_cfg_.clear_stack_frame()

        def get_op_kernel_not_found_error_str(error_cfg):
            error_msg = str(self.error_cfg_.op_kernel_not_found_error())
            error_msg = error_msg.replace("\\", "")
            error_msg = error_msg.replace("op_kernels_not_found_debug_str:", "")
            error_msg = "\n".join(
                [e.strip()[1:-1] for e in error_msg.strip().split("\n")]
            )

            return (
                "\n\nFailure messages of registered kernels for current Op node: \n%s"
                % error_msg
            )

        def get_multiple_op_kernels_matched_error_str(error_cfg):
            error_msg = str(self.error_cfg_.multiple_op_kernels_matched_error())
            error_msg = error_msg.replace("\\", "")
            error_msg = error_msg.replace("matched_op_kernels_debug_str:", "")
            error_msg = "\n".join(
                [e.strip()[1:-1] for e in error_msg.strip().split("\n")]
            )

            return (
                "\n\nThere exists multiple registered kernel candidates for current Op node: \n%s"
                % error_msg
            )

    def __str__(self):

        ret = (
            "\n\nerror msg: \n\n\033[1;31m%s\033[0m" % str(self.error_summary_).strip()
        )

        if error_cfg_.has_op_kernel_not_found_error():
            ret += self.get_op_kernel_not_found_error_str(self.error_cfg_)
            self.error_cfg_.clear_op_kernel_not_found_error()
        elif error_cfg_.multiple_op_kernels_matched_error():
            ret += self.get_multiple_op_kernels_matched_error_str(self.error_cfg_)
            self.error_cfg_.clear_multiple_op_kernels_matched_error()

        ret += "\n%s" % str(self.error_cfg_)
        ret += "\n%s" % str(self.msg_).strip()

        return ret
