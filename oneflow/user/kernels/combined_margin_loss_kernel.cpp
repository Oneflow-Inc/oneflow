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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/math_unary_elementwise_func.h"

namespace oneflow {

namespace {

class CombinedMarginLossOpKernelState final : public user_op::OpKernelState {
 public:
  CombinedMarginLossOpKernelState(int64_t lower, int64_t upper) : lower_(lower), upper_(upper) {}
  ~CombinedMarginLossOpKernelState() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};

std::shared_ptr<user_op::OpKernelState> CreateCombinedMarginLossOpKernelState(
    user_op::KernelInitContext* ctx, const std::string& in_arg_name) {
  const SbpParallel& in_sbp = ctx->SbpParallel4ArgNameAndIndex(in_arg_name, 0);
  if (in_sbp.has_split_parallel() && in_sbp.split_parallel().axis() == 1
      && ctx->parallel_ctx().parallel_num() > 1) {
    CHECK(ctx->SbpParallel4ArgNameAndIndex("label", 0).has_broadcast_parallel());
    const user_op::TensorDesc* in_logical_desc =
        ctx->LogicalTensorDesc4ArgNameAndIndex(in_arg_name, 0);
    const auto depth = ctx->Attr<int64_t>("depth");
    CHECK_EQ(depth, in_logical_desc->shape().At(1));
    BalancedSplitter bs(depth, ctx->parallel_ctx().parallel_num());
    return std::make_shared<CombinedMarginLossOpKernelState>(
        bs.At(ctx->parallel_ctx().parallel_id()).begin(),
        bs.At(ctx->parallel_ctx().parallel_id()).end());
  } else {
    return std::shared_ptr<user_op::OpKernelState>(nullptr);
  }
}

}  // namespace

template<typename T, typename K>
class CombinedMarginLossCpuKernel final : public user_op::OpKernel {
 public:
  CombinedMarginLossCpuKernel() = default;
  ~CombinedMarginLossCpuKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateCombinedMarginLossOpKernelState(ctx, "x");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const T* x_ptr = x->dptr<T>();
    const K* label_ptr = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<K>();
    T* y_ptr = ctx->Tensor4ArgNameAndIndex("y", 0)->mut_dptr<T>();
    T* theta_ptr = ctx->Tensor4ArgNameAndIndex("theta", 0)->mut_dptr<T>();
    const float m1 = ctx->Attr<float>("m1");
    const float m2 = ctx->Attr<float>("m2");
    const float m3 = ctx->Attr<float>("m3");
    int64_t lower_bound = 0;
    if (state != nullptr) {
      auto* kernel_state = dynamic_cast<CombinedMarginLossOpKernelState*>(state);
      CHECK_NOTNULL(kernel_state);
      CHECK_EQ(x->shape().Count(1), kernel_state->upper() - kernel_state->lower());
      lower_bound = kernel_state->lower();
    }
    const int64_t num_classes = x->shape().Count(1);
    FOR_RANGE(int32_t, i, 0, x->shape().elem_cnt()) {
      const int32_t row_id = i / num_classes;
      const int32_t col_id = i - row_id * num_classes;
      const T in_data = x_ptr[i];
      T out_data = in_data;
      K label = label_ptr[row_id] - lower_bound;
      if (label == col_id) {
        const T theta_data = AcosFunctor<T>::Forward(in_data);
        out_data = CosFunctor<T>::Forward(theta_data * static_cast<T>(m1) + static_cast<T>(m2))
                   - static_cast<T>(m3);
        theta_ptr[row_id] = theta_data;
      } else if ((label < 0 || label >= num_classes) && col_id == 0) {
        theta_ptr[row_id] = 0;
      }
      y_ptr[i] = out_data;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_COMBINED_MARGIN_LOSS_CPU_KERNEL(in_type, indices_type)               \
  REGISTER_USER_KERNEL("combined_margin_loss")                                        \
      .SetCreateFn<CombinedMarginLossCpuKernel<OF_PP_PAIR_FIRST(in_type),             \
                                               OF_PP_PAIR_FIRST(indices_type)>>()     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                             \
                       & (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(in_type)) \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(indices_type)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_COMBINED_MARGIN_LOSS_CPU_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

template<typename T, typename K>
class CombinedMarginLossGradCpuKernel final : public user_op::OpKernel {
 public:
  CombinedMarginLossGradCpuKernel() = default;
  ~CombinedMarginLossGradCpuKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateCombinedMarginLossOpKernelState(ctx, "dy");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const T* dy_ptr = dy->dptr<T>();
    const K* label_ptr = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<K>();
    const T* theta_ptr = ctx->Tensor4ArgNameAndIndex("theta", 0)->dptr<T>();
    T* dx_ptr = ctx->Tensor4ArgNameAndIndex("dx", 0)->mut_dptr<T>();
    const float m1 = ctx->Attr<float>("m1");
    const float m2 = ctx->Attr<float>("m2");
    int64_t lower_bound = 0;
    if (state != nullptr) {
      auto* kernel_state = dynamic_cast<CombinedMarginLossOpKernelState*>(state);
      CHECK_NOTNULL(kernel_state);
      CHECK_EQ(dy->shape().Count(1), kernel_state->upper() - kernel_state->lower());
      lower_bound = kernel_state->lower();
    }

    const int64_t num_classes = dy->shape().Count(1);
    FOR_RANGE(int32_t, i, 0, dy->shape().elem_cnt()) {
      const int32_t row_id = i / num_classes;
      const int32_t col_id = i - row_id * num_classes;
      K label = label_ptr[row_id] - lower_bound;
      const T dy_data = dy_ptr[i];
      const T theta_data = theta_ptr[row_id];
      T dx_data = dy_data;
      if (label == col_id) {
        dx_data = dy_data
                  * SinFunctor<T>::Forward(theta_data * static_cast<T>(m1) + static_cast<T>(m2))
                  * static_cast<T>(m1) / SinFunctor<T>::Forward(theta_data);
      }
      dx_ptr[i] = dx_data;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_COMBINED_MARGIN_LOSS_GRAD_CPU_KERNEL(dy_type, indices_type)           \
  REGISTER_USER_KERNEL("combined_margin_loss_grad")                                    \
      .SetCreateFn<CombinedMarginLossGradCpuKernel<OF_PP_PAIR_FIRST(dy_type),          \
                                                   OF_PP_PAIR_FIRST(indices_type)>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("dy", 0) == OF_PP_PAIR_SECOND(dy_type)) \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(indices_type)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_COMBINED_MARGIN_LOSS_GRAD_CPU_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
