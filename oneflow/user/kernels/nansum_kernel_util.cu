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
#include "oneflow/core/framework/user_op_hob.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
__global__ void NanSumKernel(const int64_t num_elements, const T* input, T* output) {
  const T zero = GetZeroVal<T>();
  CUDA_1D_KERNEL_LOOP(i, num_elements) { output[i] = isnan(input[i]) ? T{0.} : input[i]; }
}

template<typename T>
struct NanSumKernelUtil {
  static void Forward(ep::Stream* stream, const int64_t num_elements, const T* input, T* output) {
    NanSumKernel<<<BlocksNum4ThreadsNum(num_elements), kCudaThreadsNumPerBlock, 0,
                   stream->As<ep::CudaStream>()->cuda_stream()>>>(num_elements, input, output);
  }
};

template<typename T>
class CudaNanSumKernel final : public user_op::OpKernel {
 public:
  CudaNanSumKernel() = default;
  ~CudaNanSumKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t elements = in->shape_view().elem_cnt();
    NanSumKernelUtil<T>::Forward(ctx->stream(), elements, in->dptr<T>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_NANSUM_KERNELS(dtype)                                 \
  REGISTER_USER_KERNEL("replace_nansum")                               \
      .SetCreateFn<CudaNanSumKernel<dtype>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value));

REGISTER_NANSUM_KERNELS(float)
REGISTER_NANSUM_KERNELS(double)

}  // namespace oneflow
