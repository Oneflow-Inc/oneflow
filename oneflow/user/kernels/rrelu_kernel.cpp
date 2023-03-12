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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

namespace {

template<typename T, typename V>
static T uniform_real(V val, T from, T to) {
  constexpr auto MASK =
      static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
  constexpr auto DIVISOR =
      static_cast<T>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
  T x = (val & MASK) * DIVISOR;
  return (x * (to - from) + from);
}

static uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

}  // namespace

template<typename T>
class CpuRReluKernel final : public user_op::OpKernel {
 public:
  CpuRReluKernel() = default;
  ~CpuRReluKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCPU));
    generator->set_current_seed(CHECK_JUST(
        GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"), {"output", 0})));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
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

    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    auto cpu_gen = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
    std::lock_guard<std::mutex> lock(cpu_gen->mutex_);
    one::pytorch_mt19937_engine& engine = cpu_gen->torch_engine();

    FOR_RANGE(int64_t, i, 0, size) {
      if (*(in_ptr + i) >= 0) {
        noise_ptr[i] = 1;
        out_ptr[i] = in_ptr[i];
      } else {
        uint32_t random1 = engine();
        uint32_t random2 = engine();
        uint64_t rand_unit = make64BitsFrom32Bits(random1, random2);
        T uniform_sample = uniform_real(rand_unit, lower, upper);
        noise_ptr[i] = uniform_sample;
        out_ptr[i] = in_ptr[i] * uniform_sample;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_RRelu_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("rrelu").SetCreateFn<CpuRReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                  \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_CPU_RRelu_KERNEL(float);
REGISTER_CPU_RRelu_KERNEL(double);

}  // namespace oneflow
