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
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {
template<DeviceType device_type, typename T>
struct UnaryElemwiseXpuFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* out, const T* in,
                  std::function<T(T)> functor);
};

template<typename T>
struct UnaryElemwiseXpuFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* out, const T* in,
                  std::function<T(T)> functor) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = functor(in[i]); }
  }
};

template<DeviceType device_type, typename FunctorT, typename T>
class UnaryElemwiseXpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnaryElemwiseXpuKernel);
  UnaryElemwiseXpuKernel(
      const std::string& input_name, const std::string& output_name,
      std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn)
      : input_name(input_name), output_name(output_name), FunctorCreateFn(FunctorCreateFn) {}

  std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn;  // The functor

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex(input_name, 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex(output_name, 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const int64_t elem_cnt = in_tensor->shape().elem_cnt();

    UnaryElemwiseXpuFunctor<device_type, T>()(ctx->device_ctx(), elem_cnt, out_ptr, in_ptr,
                                              FunctorCreateFn(ctx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::string input_name;
  std::string output_name;
};

}  // namespace oneflow
#endif  // _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
