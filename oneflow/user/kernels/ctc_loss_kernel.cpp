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

template<DeviceType device_type, typename T, typename IDX>
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
    const int* targets_ptr = targets->dptr<int>();
    const IDX* input_lengths_ptr = input_lengths->dptr<IDX>();
    const IDX* target_lengths_ptr = target_lengths->dptr<IDX>();
    const int blank = ctx->Attr<int>("blank");
    const bool zero_infinity = ctx->Attr<bool>("zero_infinity");
    IDX max_input_length = log_probs->shape().At(0);
    IDX batch_size = log_probs->shape().At(1);
    IDX num_labels = log_probs->shape().At(2);
    IDX max_target_length = targets->shape().At(1);
    CHECK_EQ(targets->shape().At(0), batch_size);
    CHECK_EQ(input_lengths->shape().At(0), batch_size);
    CHECK_EQ(target_lengths->shape().At(0), batch_size);
    CHECK_GE(blank, 0);
    CHECK_LT(blank, num_labels);
    // for (IDX b = 0; b < batch_size; b++) { CHECK_GE(max_input_length, input_lengths_ptr[b]); }
    // for (IDX b = 0; b < batch_size; b++) { CHECK_GE(max_target_length, target_lengths_ptr[b]); }
    NdIndexOffsetHelper<IDX, 3> input_helper(max_input_length, batch_size, num_labels);
    NdIndexOffsetHelper<IDX, 3> alpha_helper(batch_size, max_input_length,
                                             2 * max_target_length + 1);
    T* loss_ptr = loss->mut_dptr<T>();
    T* alpha_ptr = alpha->mut_dptr<T>();
    CtcLossKernelUtil<device_type, T, IDX>::CtcLossForward(
        ctx->device_ctx(), batch_size, log_probs_ptr, targets_ptr, input_lengths_ptr,
        target_lengths_ptr, alpha_ptr, loss_ptr, input_helper, alpha_helper, max_target_length,
        blank, zero_infinity);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CTC_LOSS_KERNEL(device, dtype, idx_dtype)                                        \
  REGISTER_USER_KERNEL("ctc_loss")                                                                \
      .SetCreateFn<CtcLossKernel<device, OF_PP_PAIR_FIRST(dtype), OF_PP_PAIR_FIRST(idx_dtype)>>() \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceTag() == device)                                                     \
          & (user_op::HobDataType("log_probs", 0) == OF_PP_PAIR_SECOND(dtype))                    \
          & (user_op::HobDataType("input_lengths", 0) == OF_PP_PAIR_SECOND(idx_dtype)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CTC_LOSS_KERNEL, DEVICE_TYPE_SEQ,
                                 OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat), INDEX_DATA_TYPE_SEQ)

#undef REGISTER_CTC_LOSS_KERNEL

template<DeviceType device_type, typename T, typename IDX>
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
    user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);

    const T* grad_out_ptr = grad_out->dptr<T>();
    const T* loss_ptr = loss->dptr<T>();
    const T* alpha_ptr = alpha->dptr<T>();
    const T* log_probs_ptr = log_probs->dptr<T>();
    const int* targets_ptr = targets->dptr<int>();
    const IDX* input_lengths_ptr = input_lengths->dptr<IDX>();
    const IDX* target_lengths_ptr = target_lengths->dptr<IDX>();
    const int blank = ctx->Attr<int>("blank");
    const bool zero_infinity = ctx->Attr<bool>("zero_infinity");

    IDX max_input_length = log_probs->shape().At(0);
    IDX batch_size = log_probs->shape().At(1);
    IDX num_labels = log_probs->shape().At(2);
    IDX max_target_length = targets->shape().At(1);
    CHECK_EQ(targets->shape().At(0), batch_size);
    CHECK_EQ(input_lengths->shape().At(0), batch_size);
    CHECK_EQ(target_lengths->shape().At(0), batch_size);
    CHECK_GE(blank, 0);
    CHECK_LT(blank, num_labels);
    // for (IDX b = 0; b < batch_size; b++) { CHECK_GE(max_input_length, input_lengths_ptr[b]); }
    // for (IDX b = 0; b < batch_size; b++) { CHECK_GE(max_target_length, target_lengths_ptr[b]); }
    NdIndexOffsetHelper<IDX, 3> input_helper(max_input_length, batch_size, num_labels);
    NdIndexOffsetHelper<IDX, 3> beta_helper(batch_size, max_input_length,
                                            2 * max_target_length + 1);
    T* grad_ptr = grad->mut_dptr<T>();
    T* beta_ptr = beta->mut_dptr<T>();
    CtcLossKernelUtil<device_type, T, IDX>::CtcLossBackward(
        ctx->device_ctx(), grad_out_ptr, loss_ptr, alpha_ptr, batch_size, log_probs_ptr,
        targets_ptr, input_lengths_ptr, target_lengths_ptr, beta_ptr, grad_ptr, input_helper,
        beta_helper, max_input_length, max_target_length, num_labels, blank, zero_infinity);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CTC_LOSS_BACKWARD_KERNEL(device, dtype, idx_dtype)                          \
  REGISTER_USER_KERNEL("ctc_loss_grad")                                                      \
      .SetCreateFn<                                                                          \
          CtcLossGradKernel<device, OF_PP_PAIR_FIRST(dtype), OF_PP_PAIR_FIRST(idx_dtype)>>() \
      .SetIsMatchedHob(                                                                      \
          (user_op::HobDeviceTag() == device)                                                \
          & (user_op::HobDataType("log_probs", 0) == OF_PP_PAIR_SECOND(dtype))               \
          & (user_op::HobDataType("input_lengths", 0) == OF_PP_PAIR_SECOND(idx_dtype)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CTC_LOSS_BACKWARD_KERNEL, DEVICE_TYPE_SEQ,
                                 OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat), INDEX_DATA_TYPE_SEQ)

#undef REGISTER_CTC_LOSS_BACKWARD_KERNEL

}  // namespace oneflow
