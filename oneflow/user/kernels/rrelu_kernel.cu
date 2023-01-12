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
#include "oneflow/core/kernel/random_generator.h"
namespace oneflow {
namespace {

template<typename T>
__global__ void compute_rrelu(const T* in, T* out, T* noise_data, int64_t elem_cnt, const T lower,
                              const T upper, const T range) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    if (in[i] <= static_cast<T>(0.0)) {
      T temp = noise_data[i] * range + lower;

      noise_data[i] = temp;

      out[i] = in[i] * temp;

    } else {
      noise_data[i] = static_cast<T>(1.0);

      out[i] = in[i];
    }
  }
}
}  // namespace

template<typename T>
class CudaRReluKernel final : public user_op::OpKernel {
 public:
  CudaRReluKernel() = default;
  ~CudaRReluKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const int64_t size = in->shape_view().elem_cnt();
    if (size == 0) return;
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* noise_data = ctx->Tensor4ArgNameAndIndex("noise_data", 0);
    const T& lower = ctx->Attr<float>("lower");
    const T& upper = ctx->Attr<float>("upper");

    T* out_ptr = out->mut_dptr<T>();
    T* noise_ptr = noise_data->mut_dptr<T>();
    const T* in_ptr = in->dptr<T>();
    
    RandomGenerator<DeviceType::kCUDA> gen(NewRandomSeed(), ctx->stream());
    gen.Uniform<T>(size, noise_ptr);
    T range = upper - lower;
    
    RUN_CUDA_KERNEL(compute_rrelu<T>, ctx->stream(), size, in_ptr, out_ptr, noise_ptr, size, lower,
                    upper, range);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_CUDA_RRELU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("rrelu").SetCreateFn<CudaRReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                  \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_RRELU_KERNEL(float)
REGISTER_CUDA_RRELU_KERNEL(double)
}  // namespace oneflow