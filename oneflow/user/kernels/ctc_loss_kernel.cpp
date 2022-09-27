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
#include "oneflow/user/kernels/ctc_loss_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename TARGET, typename IDX>
class CtcLossKernel final : public user_op::OpKernel {
 public:
  CtcLossKernel() = default;
  ~CtcLossKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* log_probs = ctx->Tensor4ArgNameAndIndex("log_probs", 0);
    const user_op::Tensor* targets = ctx->Tensor4ArgNameAndIndex("targets", 0);
    const user_op::Tensor* input_lengths = ctx->Tensor4ArgNameAndIndex("input_lengths", 0);
    const user_op::Tensor* target_lengths = ctx->Tensor4ArgNameAndIndex("target_lengths", 0);
    user_op::Tensor* loss = ctx->Tensor4ArgNameAndIndex("loss", 0);
    user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);

    const T* log_probs_ptr = log_probs->dptr<T>();
    const TARGET* targets_ptr = targets->dptr<TARGET>();
    const IDX* input_lengths_ptr = input_lengths->dptr<IDX>();
    const IDX* target_lengths_ptr = target_lengths->dptr<IDX>();
    const int64_t blank = ctx->Attr<int64_t>("blank");
    const int64_t max_input_length = log_probs->shape_view().At(0);
    const int64_t batch_size = log_probs->shape_view().At(1);
    const int64_t num_labels = log_probs->shape_view().At(2);
    const int64_t max_target_length = ctx->Attr<int64_t>("max_target_length");
    const int32_t targets_ndim = targets->shape_view().NumAxes();

    NdIndexOffsetHelper<int64_t, 3> input_helper(max_input_length, batch_size, num_labels);
    NdIndexOffsetHelper<int64_t, 3> alpha_helper(batch_size, max_input_length,
                                                 2 * max_target_length + 1);
    T* loss_ptr = loss->mut_dptr<T>();
    T* alpha_ptr = alpha->mut_dptr<T>();
    CtcLossKernelUtil<device_type, T, TARGET, IDX>::CtcLossForward(
        ctx->stream(), log_probs_ptr, targets_ptr, input_lengths_ptr, target_lengths_ptr, alpha_ptr,
        loss_ptr, input_helper, alpha_helper, batch_size, max_input_length, max_target_length,
        blank, targets_ndim);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CTC_LOSS_KERNEL(device, dtype, target_type, idx_dtype)                          \
  REGISTER_USER_KERNEL("ctc_loss")                                                               \
      .SetCreateFn<CtcLossKernel<device, OF_PP_PAIR_FIRST(dtype), OF_PP_PAIR_FIRST(target_type), \
                                 OF_PP_PAIR_FIRST(idx_dtype)>>()                                 \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == device)                                                   \
          && (user_op::HobDataType("log_probs", 0) == OF_PP_PAIR_SECOND(dtype))                  \
          && (user_op::HobDataType("targets", 0) == OF_PP_PAIR_SECOND(target_type))              \
          && (user_op::HobDataType("input_lengths", 0) == OF_PP_PAIR_SECOND(idx_dtype)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CTC_LOSS_KERNEL, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

template<DeviceType device_type, typename T, typename TARGET, typename IDX>
class CtcLossGradKernel final : public user_op::OpKernel {
 public:
  CtcLossGradKernel() = default;
  ~CtcLossGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* grad_out = ctx->Tensor4ArgNameAndIndex("grad_out", 0);
    const user_op::Tensor* loss = ctx->Tensor4ArgNameAndIndex("loss", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* log_probs = ctx->Tensor4ArgNameAndIndex("log_probs", 0);
    const user_op::Tensor* targets = ctx->Tensor4ArgNameAndIndex("targets", 0);
    const user_op::Tensor* input_lengths = ctx->Tensor4ArgNameAndIndex("input_lengths", 0);
    const user_op::Tensor* target_lengths = ctx->Tensor4ArgNameAndIndex("target_lengths", 0);
    user_op::Tensor* grad = ctx->Tensor4ArgNameAndIndex("grad", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const T* grad_out_ptr = grad_out->dptr<T>();
    const T* loss_ptr = loss->dptr<T>();
    const T* alpha_ptr = alpha->dptr<T>();
    const T* log_probs_ptr = log_probs->dptr<T>();
    const TARGET* targets_ptr = targets->dptr<TARGET>();
    const IDX* input_lengths_ptr = input_lengths->dptr<IDX>();
    const IDX* target_lengths_ptr = target_lengths->dptr<IDX>();
    const int64_t blank = ctx->Attr<int64_t>("blank");
    const bool zero_infinity = ctx->Attr<bool>("zero_infinity");
    const int64_t batch_size = log_probs->shape_view().At(1);
    const int64_t num_labels = log_probs->shape_view().At(2);
    const int64_t max_input_length = log_probs->shape_view().At(0);
    const int64_t max_target_length = ctx->Attr<int64_t>("max_target_length");
    const int32_t targets_ndim = targets->shape_view().NumAxes();

    NdIndexOffsetHelper<int64_t, 3> input_helper(max_input_length, batch_size, num_labels);
    NdIndexOffsetHelper<int64_t, 3> beta_helper(batch_size, max_input_length,
                                                2 * max_target_length + 1);
    T* grad_ptr = grad->mut_dptr<T>();
    T* beta_ptr = tmp_buffer->mut_dptr<T>();
    CtcLossKernelUtil<device_type, T, TARGET, IDX>::CtcLossBackward(
        ctx->stream(), grad_out_ptr, loss_ptr, alpha_ptr, log_probs_ptr, targets_ptr,
        input_lengths_ptr, target_lengths_ptr, beta_ptr, grad_ptr, input_helper, beta_helper,
        batch_size, max_input_length, max_target_length, num_labels, blank, zero_infinity,
        targets_ndim);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CTC_LOSS_BACKWARD_KERNEL(device, dtype, target_type, idx_dtype)            \
  REGISTER_USER_KERNEL("ctc_loss_grad")                                                     \
      .SetCreateFn<                                                                         \
          CtcLossGradKernel<device, OF_PP_PAIR_FIRST(dtype), OF_PP_PAIR_FIRST(target_type), \
                            OF_PP_PAIR_FIRST(idx_dtype)>>()                                 \
      .SetIsMatchedHob(                                                                     \
          (user_op::HobDeviceType() == device)                                              \
          && (user_op::HobDataType("log_probs", 0) == OF_PP_PAIR_SECOND(dtype))             \
          && (user_op::HobDataType("targets", 0) == OF_PP_PAIR_SECOND(target_type))         \
          && (user_op::HobDataType("input_lengths", 0) == OF_PP_PAIR_SECOND(idx_dtype)))    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const Shape& log_probs_shape = ctx->InputShape("log_probs", 0);                     \
        const int64_t max_target_length = ctx->Attr<int64_t>("max_target_length");          \
        int64_t elem_cnt =                                                                  \
            log_probs_shape.At(1) * log_probs_shape.At(0) * (2 * max_target_length + 1);    \
        return elem_cnt * sizeof(OF_PP_PAIR_FIRST(dtype));                                  \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CTC_LOSS_BACKWARD_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
