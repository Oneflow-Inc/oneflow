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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/where_kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename CondT>
class MaskedFillKernel final : public user_op::OpKernel {
 public:
  MaskedFillKernel() = default;
  ~MaskedFillKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else if (ctx->Attr<bool>("has_bool_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<bool>("bool_operand"));
    } else {
      UNIMPLEMENTED() << "The scalar in MaskedFill should be float or int.";
    }
    WhereKernelUtil<device_type, T, CondT>::WhereXScalar(
        ctx->stream(), out->shape_view().elem_cnt(), mask->dptr<CondT>(), scalar_operand,
        x->dptr<T>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MASKED_FILL_KERNEL(device_type_v, dtype_pair, ctype_pair)                   \
  REGISTER_USER_KERNEL("masked_fill")                                                        \
      .SetCreateFn<MaskedFillKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),             \
                                    OF_PP_PAIR_FIRST(ctype_pair)>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type_v)                           \
                       && (user_op::HobDataType("mask", 0) == OF_PP_PAIR_SECOND(ctype_pair)) \
                       && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MASKED_FILL_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MASKED_FILL_KERNEL, (DeviceType::kCUDA),
                                 FLOAT16_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)
#endif

}  // namespace oneflow
