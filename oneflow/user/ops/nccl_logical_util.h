/*
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
*/
#ifndef ONEFLOW_USER_OPS_NCCL_LOGICAL_UTIL_H_
#define ONEFLOW_USER_OPS_NCCL_LOGICAL_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

inline Maybe<void> GetNcclLogicalNdSbpFromAttr(user_op::InferNdSbpFnContext* ctx,
                                               const std::string& attr_name, cfg::NdSbp* nd_sbp) {
  const auto& sbp_str_list = ctx->user_op_conf().attr<std::vector<std::string>>(attr_name);

  if (!ParseNdSbpFromStringList(sbp_str_list, nd_sbp)) {
    std::ostringstream err;
    err << "invalid " << attr_name << ": [";
    for (size_t i = 0; i < sbp_str_list.size(); ++i) {
      const auto& sbp_str = sbp_str_list[i];
      if (i == 0) {
        err << sbp_str;
      } else {
        err << ", " << sbp_str;
      }
    }
    err << "] for " << ctx->user_op_conf().op_type_name();
    return Error::RuntimeError() << err.str();
  }

  return Maybe<void>::Ok();
}

}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_NCCL_LOGICAL_UTIL_H_
