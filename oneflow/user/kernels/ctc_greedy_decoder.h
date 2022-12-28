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
#ifndef _ONEFLOW_USER_KERNELS_CTC_GREEDY_DECODER_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_CTC_GREEDY_DECODER_KERNEL_H_
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {
template<DeviceType device_type, typename T>
struct CTCGreedyDecoderFunctor final {
  void operator()(ep::Stream* stream, int64_t* decoded_ptr, T* neg_sum_logits_ptr,
                  const T* log_probs_ptr, const int64_t* input_lengths_ptr,
                  const bool merge_repeated, const int64_t max_input_length,
                  const int64_t batch_size, const int64_t num_labels);
};

}  // namespace

template<DeviceType device_type, typename T>
class CTCGreedyDecoderKernel final : public user_op::OpKernel {
 public:
  CTCGreedyDecoderKernel() = default;
  ~CTCGreedyDecoderKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* log_probs = ctx->Tensor4ArgNameAndIndex("log_probs", 0);
    const user_op::Tensor* input_lengths = ctx->Tensor4ArgNameAndIndex("input_lengths", 0);
    user_op::Tensor* decoded = ctx->Tensor4ArgNameAndIndex("decoded", 0);
    user_op::Tensor* neg_sum_logits = ctx->Tensor4ArgNameAndIndex("neg_sum_logits", 0);
    const T* log_probs_ptr = log_probs->dptr<T>();
    const int64_t* input_lengths_ptr = input_lengths->dptr<int64_t>();
    const bool merge_repeated = ctx->Attr<bool>("merge_repeated");
    const int64_t max_input_length = log_probs->shape_view().At(0);
    const int64_t batch_size = log_probs->shape_view().At(1);
    const int64_t num_labels = log_probs->shape_view().At(2);
    CHECK_EQ(batch_size, input_lengths->shape_view().At(0));
    int64_t* decoded_ptr = decoded->mut_dptr<int64_t>();
    T* neg_sum_logits_ptr = neg_sum_logits->mut_dptr<T>();

    CTCGreedyDecoderFunctor<device_type, T>()(ctx->stream(), decoded_ptr, neg_sum_logits_ptr,
                                              log_probs_ptr, input_lengths_ptr, merge_repeated,
                                              max_input_length, batch_size, num_labels);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CTC_GREEDY_DECODER_KERNELS(device, dtype)  \
  REGISTER_USER_KERNEL("ctc_greedy_decoder")                \
      .SetCreateFn<CTCGreedyDecoderKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("log_probs", 0) == GetDataType<dtype>::value));

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_CTC_GREEDY_DECODER_KERNEL_H_
