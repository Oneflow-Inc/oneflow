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

template<typename ContextT, typename AttrT>
struct AttrFromContext {
  const AttrT& operator()(ContextT*, const std::string&);
};

template<typename AttrT>
struct AttrFromContext<user_op::InferNdSbpFnContext, AttrT> {
  const AttrT& operator()(user_op::InferNdSbpFnContext* ctx, const std::string& attr_name) {
    return ctx->user_op_conf().template attr<AttrT>(attr_name);
  }
};

template<typename AttrT>
struct AttrFromContext<user_op::KernelInitContext, AttrT> {
  const AttrT& operator()(user_op::KernelInitContext* ctx, const std::string& attr_name) {
    return ctx->Attr<AttrT>(attr_name);
  }
};

template<typename AttrT>
struct AttrFromContext<user_op::InferContext, AttrT> {
  const AttrT& operator()(user_op::InferContext* ctx, const std::string& attr_name) {
    return ctx->Attr<AttrT>(attr_name);
  }
};

template<typename ContextT>
struct OpTypeNameFromContext {
  const std::string& operator()(ContextT*);
};

template<>
struct OpTypeNameFromContext<user_op::InferNdSbpFnContext> {
  const std::string& operator()(user_op::InferNdSbpFnContext* ctx) {
    return ctx->user_op_conf().op_type_name();
  }
};

template<>
struct OpTypeNameFromContext<user_op::KernelInitContext> {
  const std::string& operator()(user_op::KernelInitContext* ctx) { return ctx->op_type_name(); }
};

template<>
struct OpTypeNameFromContext<user_op::InferContext> {
  const std::string& operator()(user_op::InferContext* ctx) { return ctx->op_type_name(); }
};

template<typename ContextT>
Maybe<void> GetNcclLogicalNdSbpFromAttr(ContextT* ctx, const std::string& attr_name,
                                        NdSbp* nd_sbp) {
  const auto& sbp_str_list = AttrFromContext<ContextT, std::vector<std::string>>()(ctx, attr_name);

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
    err << "] for " << OpTypeNameFromContext<ContextT>()(ctx);
    return Error::RuntimeError() << err.str();
  }

  return Maybe<void>::Ok();
}

}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_NCCL_LOGICAL_UTIL_H_
