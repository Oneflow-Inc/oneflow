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
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_GPU_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_GPU_KERNEL_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

template<DeviceType device_type, typename FunctorT, typename T>
class UnaryElemwiseGpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnaryElemwiseGpuKernel);
  UnaryElemwiseGpuKernel(
      const std::string& input_name, const std::string& output_name,
      std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn)
      : input_name(input_name), output_name(output_name), FunctorCreateFn(FunctorCreateFn) {}

  std::string input_name = "in";    // The name for the input tensor
  std::string output_name = "out";  // The name for the output tensor

  std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn;  // The functor

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex(input_name, 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex(output_name, 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const int64_t elem_cnt = in_tensor->shape().elem_cnt();

    FunctorT functor = FunctorCreateFn(ctx);
    OF_CUDA_CHECK(oneflow::cuda::elementwise::Unary(functor, elem_cnt, out_ptr, in_ptr,
                                                    ctx->device_ctx()->cuda_stream()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace oneflow
#endif
