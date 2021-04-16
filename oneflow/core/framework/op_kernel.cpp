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
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {

namespace user_op {

void OpKernel::InferShape(KernelInferContext* ctx) const {
  InferContext* op_infer_ctx = ctx->MutOpInferContext();
  CHECK_NOTNULL(op_infer_ctx);
  ctx->GetOpInferFn()(op_infer_ctx);
  for (const auto& arg_pair : ctx->outputs()) {
    const Shape& shape = *op_infer_ctx->Shape4ArgNameAndIndex(arg_pair.first, arg_pair.second);
    auto* mut_shape_view = ctx->MutShapeView4ArgNameAndIndex(arg_pair.first, arg_pair.second);
    if (mut_shape_view) { mut_shape_view->set_shape(shape); }
  }
}

#define KERNEL_CONTETX_ATTR_MEMBER_FUNC(field, cpp_type, attr_type)                          \
  template<>                                                                                 \
  const cpp_type& KernelCreateContext::attr<cpp_type>(const std::string& attr_name) const {  \
    auto it = attrs_.find(attr_name);                                                        \
    if (it != attrs_.end()) {                                                                \
      return std::dynamic_pointer_cast<TypedAttrVal<cpp_type>>(it->second)->val();           \
    } else {                                                                                 \
      LOG(FATAL) << "Cannot find the attr: " << attr_name                                    \
                 << " with AttrType: " << static_cast<int32_t>(attr_type);                   \
    }                                                                                        \
  }                                                                                          \
  template<>                                                                                 \
  const cpp_type& KernelInitContext::attr<cpp_type>(const std::string& attr_name) const {    \
    auto it = attrs_.find(attr_name);                                                        \
    if (it != attrs_.end()) {                                                                \
      return std::dynamic_pointer_cast<TypedAttrVal<cpp_type>>(it->second)->val();           \
    } else {                                                                                 \
      LOG(FATAL) << "Cannot find the attr: " << attr_name                                    \
                 << " with AttrType: " << static_cast<int32_t>(attr_type);                   \
    }                                                                                        \
  }                                                                                          \
  template<>                                                                                 \
  const cpp_type& KernelInferContext::attr<cpp_type>(const std::string& attr_name) const {   \
    auto it = attrs_.find(attr_name);                                                        \
    if (it != attrs_.end()) {                                                                \
      return std::dynamic_pointer_cast<TypedAttrVal<cpp_type>>(it->second)->val();           \
    } else {                                                                                 \
      LOG(FATAL) << "Cannot find the attr: " << attr_name                                    \
                 << " with AttrType: " << static_cast<int32_t>(attr_type);                   \
    }                                                                                        \
  }                                                                                          \
  template<>                                                                                 \
  const cpp_type& KernelComputeContext::attr<cpp_type>(const std::string& attr_name) const { \
    auto it = attrs_.find(attr_name);                                                        \
    if (it != attrs_.end()) {                                                                \
      return std::dynamic_pointer_cast<TypedAttrVal<cpp_type>>(it->second)->val();           \
    } else {                                                                                 \
      LOG(FATAL) << "Cannot find the attr: " << attr_name                                    \
                 << " with AttrType: " << static_cast<int32_t>(attr_type);                   \
    }                                                                                        \
  }

OF_PP_FOR_EACH_TUPLE(KERNEL_CONTETX_ATTR_MEMBER_FUNC, ATTR_SEQ)

#undef KERNEL_CONTETX_ATTR_MEMBER_FUNC

}  // namespace user_op

}  // namespace oneflow
