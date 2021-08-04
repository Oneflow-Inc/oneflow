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

#include "oneflow/user/kernels/randint_kernel.h"
#include <curand.h>
#include <curand_kernel.h>
namespace oneflow {
__global__ void GenValues(int64_t* a, const int64_t low, const int64_t high, int32_t n,
                          curandState* state) {
  XPU_1D_KERNEL_LOOP(i, n) {
    a[i] = curand(state + i) % (high - low)
           + low;  //@TODO:curandState only generates 32-bit random number
  }
}

class GpuRandintKernel final : public user_op::OpKernel {
 public:
  GpuRandintKernel() = default;
  ~GpuRandintKernel() = default;
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeAutoGenerator());
    generator->set_current_seed(ctx->Attr<int64_t>("seed"));
    return std::make_shared<RandintKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t* output = out->mut_dptr<int64_t>();
    auto* randint_kernel_state = dynamic_cast<RandintKernelState*>(state);
    CHECK_NOTNULL(randint_kernel_state);
    const auto& generator = randint_kernel_state->generator();
    const auto& gpu_generator = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());
    CHECK_NOTNULL(generator);

    int32_t block_num = gpu_generator->max_block_num();
    int32_t thread_num = gpu_generator->max_thread_num();
    curandState* curand_states = gpu_generator->curand_states();

    const int32_t n = out->shape().elem_cnt();
    const int64_t low = ctx->Attr<int64_t>("low");
    const int64_t high = ctx->Attr<int64_t>("high");
    GenValues<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                ctx->device_ctx()->cuda_stream()>>>(output, low, high, n, curand_states);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_GPU_RANDINT_KERNEL                                                \
  REGISTER_USER_KERNEL("randint").SetCreateFn<GpuRandintKernel>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu"));

REGISTER_GPU_RANDINT_KERNEL
}  // namespace oneflow