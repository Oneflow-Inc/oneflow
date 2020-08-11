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
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

template<typename T, typename K>
class BernoulliKerenl final : public user_op::OpKernel {
 public:
  BernoulliKerenl() = default;
  ~BernoulliKerenl() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    int64_t seed = GetOpKernelRandomSeed(ctx);
    return std::make_shared<OpKernelStateWrapper<std::mt19937>>(seed);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* random_generator = dynamic_cast<OpKernelStateWrapper<std::mt19937>*>(state);
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_dptr = in_blob->dptr<T>();
    K* out_dptr = out_blob->mut_dptr<K>();
    CHECK_EQ(GetDataType<T>(), in_blob->data_type());
    CHECK_EQ(GetDataType<K>(), out_blob->data_type());
    CHECK_EQ(in_blob->shape().elem_cnt(), out_blob->shape().elem_cnt());
    for (int32_t i = 0; i < out_blob->shape().elem_cnt(); ++i) {
      double prob = static_cast<double>(*(in_dptr + i));
      CHECK(prob >= 0.0 && prob <= 1.0);
      std::bernoulli_distribution dis(prob);
      *(out_dptr + i) = dis(*random_generator->Mutable()) ? GetOneVal<K>() : GetZeroVal<K>();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BERNOULLI_KERNEL(in_dtype_pair, out_dtype_pair)                                \
  REGISTER_USER_KERNEL("bernoulli")                                                             \
      .SetCreateFn<                                                                             \
          BernoulliKerenl<OF_PP_PAIR_FIRST(in_dtype_pair), OF_PP_PAIR_FIRST(out_dtype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                       \
                       & (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(in_dtype_pair))    \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(out_dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BERNOULLI_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
