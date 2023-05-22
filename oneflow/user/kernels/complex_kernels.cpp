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
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/elementwise_unary.h"
#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/user/kernels/elementwise_primitive_kernel.h"
#include <complex>
#ifdef WITH_CUDA
#include <cuComplex.h>
#endif  // WITH_CUDA

namespace oneflow {
namespace user_op {

#define COMPLEX_UNARY_ELEMENTWISE_PRIMITIVE_SEQ                        \
  OF_PP_MAKE_TUPLE_SEQ("conj_physical", ep::primitive::UnaryOp::kConj) \
  OF_PP_MAKE_TUPLE_SEQ("real", ep::primitive::UnaryOp::kReal)          \
  OF_PP_MAKE_TUPLE_SEQ("imag", ep::primitive::UnaryOp::kImag)

#define COMPLEX_UNARY_GRAD_ELEMENTWISE_PRIMITIVE_SEQ                   \
  OF_PP_MAKE_TUPLE_SEQ("real_grad", ep::primitive::UnaryOp::kRealGrad) \
  OF_PP_MAKE_TUPLE_SEQ("imag_grad", ep::primitive::UnaryOp::kImagGrad)

#define REGISTER_COMPLEX_KERNEL(name, UnaryOp)                                            \
  REGISTER_USER_KERNEL(name)                                                              \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "out", "x", [](user_op::KernelComputeContext* ctx) {                          \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0); \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);   \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), UnaryOp, src->data_type(), dst->data_type());       \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(UnaryOp, "out", "x"));
OF_PP_FOR_EACH_TUPLE(REGISTER_COMPLEX_KERNEL, COMPLEX_UNARY_ELEMENTWISE_PRIMITIVE_SEQ)

#define REGISTER_COMPLEX_GRAD_KERNEL(name, UnaryOp)                                        \
  REGISTER_USER_KERNEL(name)                                                               \
      .SetCreateFn([]() {                                                                  \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                 \
            "dx", "dout", [](user_op::KernelComputeContext* ctx) {                         \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);   \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dout", 0); \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(  \
                  ctx->device_type(), UnaryOp, src->data_type(), dst->data_type());        \
            });                                                                            \
      })                                                                                   \
      .SetIsMatchedHob(UnaryPrimitiveExists(UnaryOp, "dx", "dout"));

OF_PP_FOR_EACH_TUPLE(REGISTER_COMPLEX_GRAD_KERNEL, COMPLEX_UNARY_GRAD_ELEMENTWISE_PRIMITIVE_SEQ)

}  // namespace user_op
}  // namespace oneflow
