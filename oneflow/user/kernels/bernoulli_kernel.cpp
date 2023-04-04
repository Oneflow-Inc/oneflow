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
#include "oneflow/user/kernels/op_kernel_wrapper.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/kernels/random_mask_generator.h"

namespace oneflow {

template<typename T, typename K>
class BernoulliKerenl final : public user_op::OpKernel {
 public:
  BernoulliKerenl() = default;
  ~BernoulliKerenl() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCPU));
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_dptr = in_blob->dptr<T>();
    K* out_dptr = out_blob->mut_dptr<K>();
    CHECK_EQ(GetDataType<T>(), in_blob->data_type());
    CHECK_EQ(GetDataType<K>(), out_blob->data_type());
    CHECK_EQ(in_blob->shape_view().elem_cnt(), out_blob->shape_view().elem_cnt());

    auto* kernel_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    const auto& generator = kernel_state->generator();
    CHECK_NOTNULL(generator);
    const auto& cpu_generator = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());

    double p = ctx->Attr<double>("p");
    // prob != -1 means use prob instead of tensor to generate random number
    if (p != static_cast<double>(-1.0)) {
      for (int32_t i = 0; i < out_blob->shape_view().elem_cnt(); ++i) {
        std::bernoulli_distribution dis(p);
        *(out_dptr + i) = dis(cpu_generator->engine()) ? GetOneVal<K>() : GetZeroVal<K>();
      }
    } else {
      for (int32_t i = 0; i < out_blob->shape_view().elem_cnt(); ++i) {
        double prob = static_cast<double>(*(in_dptr + i));
        CHECK(prob >= 0.0 && prob <= 1.0);
        std::bernoulli_distribution dis(prob);
        *(out_dptr + i) = dis(cpu_generator->engine()) ? GetOneVal<K>() : GetZeroVal<K>();
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BERNOULLI_KERNEL(in_dtype_pair, out_dtype_pair)                                \
  REGISTER_USER_KERNEL("bernoulli")                                                             \
      .SetCreateFn<                                                                             \
          BernoulliKerenl<OF_PP_PAIR_FIRST(in_dtype_pair), OF_PP_PAIR_FIRST(out_dtype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       && (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(in_dtype_pair))   \
                       && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(out_dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BERNOULLI_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
