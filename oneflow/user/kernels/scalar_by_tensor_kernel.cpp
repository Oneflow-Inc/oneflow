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
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

enum ByScalarFunc { add, sub, mul, div };

template<ByScalarFunc by_scalar_func, DeviceType device_type, typename T>
struct ComputeScalarByTensor {
  static void DoCompute(DeviceCtx* ctx, const T* x_ptr, const T* scalar_ptr, T* y_ptr,
                        const int64_t n);
};

template<DeviceType device_type, typename T>
struct ComputeScalarByTensor<ByScalarFunc::add, device_type, T> {
  static void DoCompute(DeviceCtx* ctx, const T* x_ptr, const T* scalar_ptr, T* y_ptr,
                        const int64_t n) {
    NewKernelUtil<device_type>::AddByScalarPtr(ctx, n, x_ptr, scalar_ptr, y_ptr);
  }
};

template<DeviceType device_type, typename T>
struct ComputeScalarByTensor<ByScalarFunc::sub, device_type, T> {
  static void DoCompute(DeviceCtx* ctx, const T* x_ptr, const T* scalar_ptr, T* y_ptr,
                        const int64_t n) {
    NewKernelUtil<device_type>::SubByScalarPtr(ctx, n, x_ptr, scalar_ptr, y_ptr);
  }
};

template<DeviceType device_type, typename T>
struct ComputeScalarByTensor<ByScalarFunc::mul, device_type, T> {
  static void DoCompute(DeviceCtx* ctx, const T* x_ptr, const T* scalar_ptr, T* y_ptr,
                        const int64_t n) {
    NewKernelUtil<device_type>::MulByScalarPtr(ctx, n, x_ptr, scalar_ptr, y_ptr);
  }
};

template<DeviceType device_type, typename T>
struct ComputeScalarByTensor<ByScalarFunc::div, device_type, T> {
  static void DoCompute(DeviceCtx* ctx, const T* x_ptr, const T* scalar_ptr, T* y_ptr,
                        const int64_t n) {
    NewKernelUtil<device_type>::DivByScalarPtr(ctx, n, x_ptr, scalar_ptr, y_ptr);
  }
};

template<ByScalarFunc by_scalar_func, DeviceType device, typename T>
class ScalarAddByTensorKernel final : public user_op::OpKernel {
 public:
  ScalarAddByTensorKernel() = default;
  ~ScalarAddByTensorKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    ComputeScalarByTensor<by_scalar_func, device, T>::DoCompute(ctx->device_ctx(), x->dptr<T>(),
                                                                scalar->dptr<T>(), y->mut_dptr<T>(),
                                                                x->shape().elem_cnt());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_SCALAR_ADD_BY_TENSOR_KERNEL(scalar_by_tensor_pair, device, dtype_pair)         \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(scalar_by_tensor_pair))                                 \
      .SetCreateFn<ScalarAddByTensorKernel<OF_PP_PAIR_SECOND(scalar_by_tensor_pair), device,    \
                                           OF_PP_PAIR_FIRST(dtype_pair)>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(dtype_pair)))       \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

#define SCALAR_BY_TENSOR_SEQ                                      \
  OF_PP_MAKE_TUPLE_SEQ("scalar_add_by_tensor", ByScalarFunc::add) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_sub_by_tensor", ByScalarFunc::sub) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_mul_by_tensor", ByScalarFunc::mul) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_div_by_tensor", ByScalarFunc::div)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_ADD_BY_TENSOR_KERNEL, SCALAR_BY_TENSOR_SEQ,
                                 DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
