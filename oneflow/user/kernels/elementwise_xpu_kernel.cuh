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
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_CUH_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_CUH_
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

template<typename FunctorT, typename OutputT, typename InputA>
struct UnaryElemwiseXpuLauncher<DeviceType::kGPU, FunctorT, OutputT, InputA> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, OutputT* out, const InputA* input_a,
                  FunctorT functor) {
    OF_CUDA_CHECK(cuda::elementwise::Unary(functor, elem_cnt, out, input_a, ctx->cuda_stream()));
  }
};

template<typename FunctorT, typename OutputT, typename InputA, typename InputB>
struct BinaryElemwiseXpuLauncher<DeviceType::kGPU, FunctorT, OutputT, InputA, InputB> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, OutputT* out, const InputA* input_a,
                  const InputB* input_b, FunctorT functor) {
    OF_CUDA_CHECK(
        cuda::elementwise::Binary(functor, elem_cnt, out, input_a, input_b, ctx->cuda_stream()));
  }
};

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_CUH_
