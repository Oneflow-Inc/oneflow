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
#include "oneflow/user/kernels/elementwise_primitive_kernel.h"

namespace oneflow {
namespace user_op {
#define UTIL_OPS_SEQ                                            \
  OF_PP_MAKE_TUPLE_SEQ("isinf", ep::primitive::UnaryOp::kIsInf) \
  OF_PP_MAKE_TUPLE_SEQ("isnan", ep::primitive::UnaryOp::kIsNan) \
  OF_PP_MAKE_TUPLE_SEQ("isfinite", ep::primitive::UnaryOp::kIsFinite)

#define RISTER_UTIL_OPS(op_name, op_kind)                                                 \
  REGISTER_USER_KERNEL(op_name)                                                           \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                         \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);  \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0); \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), op_kind, src->data_type(), dst->data_type());       \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(op_kind, "out", "in"));

OF_PP_FOR_EACH_TUPLE(RISTER_UTIL_OPS, UTIL_OPS_SEQ)
#undef RISTER_UTIL_OPS
#undef UTIL_OPS_SEQ
}  // namespace user_op
}  // namespace oneflow
