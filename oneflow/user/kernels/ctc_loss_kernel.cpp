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
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* loss = ctx->Tensor4ArgNameAndIndex("loss", 0);

    const T* log_probs_ptr = log_probs->dptr<T>();
    const int* targets_ptr = targets->dptr<int>();
    const IDX* input_lengths_ptr = input_lengths->dptr<IDX>();
    const IDX* target_length_ptr = target_lengths->dptr<IDX>();
    const int blank = ctx->Attr<int>("blank");
    IDX num_labels = log_probs->shape().At(2);
    CHECK_GE(blank, 0);
    CHECK_LT(blank, num_labels);

    IDX batch_size = log_probs->shape().At(1);
    IDX max_input_length = log_probs->shape().At(0);
    for (int64_t b = 0; b < batch_size; b++) { CHECK_GE(max_input_length, input_lengths_ptr[b]); }

    IDX max_target_length = targets->shape().At(1);
    for (IDX b = 0; b < batch_size; b++) { CHECK_GE(max_target_length, target_length_ptr[b]); }
    NdIndexOffsetHelper<IDX, 3> input_helper(max_input_length, batch_size, num_labels);
    NdIndexOffsetHelper<IDX, 3> alpha_helper(batch_size, max_input_length,
                                             2 * max_target_length + 1);
    T* alpha_ptr = tmp_buffer->mut_dptr<T>();
    T* loss_ptr = loss->mut_dptr<T>();
    CtcLossKernelUtil<device_type, T, IDX>::CtcLossForward(
        ctx->device_ctx(), batch_size, log_probs_ptr, targets_ptr, input_lengths_ptr,
        target_length_ptr, alpha_ptr, loss_ptr, input_helper, alpha_helper, max_target_length,
        blank);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CTC_LOSS_KERNEL(device, dtype, idx_dtype)                                        \
  REGISTER_USER_KERNEL("ctc_loss")                                                                \
      .SetCreateFn<CtcLossKernel<device, OF_PP_PAIR_FIRST(dtype), OF_PP_PAIR_FIRST(idx_dtype)>>() \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceTag() == device)                                                     \
          & (user_op::HobDataType("log_probs", 0) == OF_PP_PAIR_SECOND(dtype))                    \
          & (user_op::HobDataType("input_lengths", 0) == OF_PP_PAIR_SECOND(idx_dtype)))           \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const Shape* log_probs_shape = ctx->Shape4ArgNameAndIndex("log_probs", 0);                \
        const Shape* targets_shape = ctx->Shape4ArgNameAndIndex("targets", 0);                    \
        int64_t elem_cnt =                                                                        \
            log_probs_shape->At(1) * log_probs_shape->At(0) * (2 * targets_shape->At(1) + 1);     \
        return elem_cnt * sizeof(OF_PP_PAIR_FIRST(dtype));                                        \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CTC_LOSS_KERNEL, DEVICE_TYPE_SEQ,
                                 OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat), INDEX_DATA_TYPE_SEQ)

#undef REGISTER_CTC_LOSS_KERNEL

}  // namespace oneflow
