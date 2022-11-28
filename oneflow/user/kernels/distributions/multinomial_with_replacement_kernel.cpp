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
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"

// NOTE(Liang Depeng): The implementation of MultinomialWithReplacementCpuKernel is modified from
//                    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/MultinomialKernel.cpp#L23
namespace oneflow {

namespace {

static size_t InferTmpSizeForCpuKernel(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("x", 0);
  int64_t n_categories = x.shape().At(x.shape().NumAxes() - 1);
  return n_categories * GetSizeOfDataType(x.data_type());
}

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
class MultinomialWithReplacementCpuKernel final : public user_op::OpKernel {
 public:
  MultinomialWithReplacementCpuKernel() = default;
  ~MultinomialWithReplacementCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCPU));
    // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    auto cpu_gen = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
    std::lock_guard<std::mutex> lock(cpu_gen->mutex_);

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const T* self_ptr = x->dptr<T>();
    int64_t* result_ptr = out->mut_dptr<int64_t>();
    /* cumulative probability distribution vector */
    T* cum_dist_ptr = tmp_buffer->mut_dptr<T>();

    int64_t n_categories = x->shape_view().At(x->shape_view().NumAxes() - 1);
    int64_t n_dist = x->shape_view().NumAxes() > 1 ? x->shape_view().At(0) : 1;
    const int32_t num_samples = ctx->Attr<int32_t>("num_samples");

    int64_t self_stride_0 = x->shape_view().NumAxes() > 1 ? x->stride().at(0) : 0;
    int64_t self_stride_1 = x->stride().at(x->shape_view().NumAxes() - 1);
    int64_t result_dist_stride_0 = out->shape_view().NumAxes() > 1 ? out->stride().at(0) : 0;
    int64_t result_dist_stride_1 = out->stride().at(out->shape_view().NumAxes() - 1);

    one::pytorch_mt19937_engine& engine = cpu_gen->torch_engine();

    for (int i = 0; i < n_dist; ++i) {
      /* Get normalized cumulative distribution from prob distribution */
      T sum = 0;
      T val;
      for (int j = 0; j < n_categories; ++j) {
        val = self_ptr[i * self_stride_0 + j * self_stride_1];
        CHECK(val >= 0) << "invalid multinomial distribution (encountering probability entry < 0)";
        CHECK(std::isfinite(val)) << "invalid multinomial distribution (encountering probability "
                                     "entry = infinity or NaN)";
        sum += val;
        cum_dist_ptr[j] = sum;
      }

      CHECK(sum > 0) << "invalid multinomial distribution (sum of probabilities <= 0)";

      /* normalize cumulative probability distribution so that last val is 1
      i.e. doesn't assume original self row sums to one */
      if ((sum > 0) || ((sum < 1.00001) && (sum > 0.99999))) {
        for (int j = 0; j < n_categories; ++j) { cum_dist_ptr[j] /= sum; }
      }

      for (int j = 0; j < num_samples; ++j) {
        /* sample a probability mass from a uniform distribution */
        // at::uniform_real_distribution<double> uniform(0, 1);
        // double uniform_sample = uniform(gen);
        uint32_t random1 = engine();
        uint32_t random2 = engine();
        uint64_t rand_unit = make64BitsFrom32Bits(random1, random2);
        double uniform_sample = uniform_real(rand_unit, 0.0, 1.0);

        // Do a binary search for the slot in which the prob falls
        // ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot]
        int left_pointer = 0;
        int right_pointer = n_categories;
        int mid_pointer = 0;
        T cum_prob;
        int sample_idx = 0;
        // Make sure the last cumulative distribution bucket sums to 1
        cum_dist_ptr[(n_categories - 1)] = 1;

        while (right_pointer - left_pointer > 0) {
          mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
          cum_prob = cum_dist_ptr[mid_pointer];
          if (cum_prob < uniform_sample) {
            left_pointer = mid_pointer + 1;
          } else {
            right_pointer = mid_pointer;
          }
        }
        sample_idx = left_pointer;

        // store in result tensor (will be incremented for lua compat by wrapper)
        result_ptr[i * result_dist_stride_0 + j * result_dist_stride_1] = sample_idx;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MULTINOMIAL_WITH_REPLACEMENT_CPU_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("multinomial_with_replacement")                                 \
      .SetCreateFn<MultinomialWithReplacementCpuKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                  \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTmpSizeForCpuKernel);

REGISTER_MULTINOMIAL_WITH_REPLACEMENT_CPU_KERNEL(float)
REGISTER_MULTINOMIAL_WITH_REPLACEMENT_CPU_KERNEL(double)

}  // namespace oneflow