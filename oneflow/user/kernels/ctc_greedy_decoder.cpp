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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/ctc_greedy_decoder.h"

namespace oneflow {
namespace {

template<typename T>
struct CTCGreedyDecoderFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, int64_t* decoded_ptr, T* neg_sum_logits_ptr,
                  const T* log_probs_ptr, const int64_t* input_lengths_ptr,
                  const bool merge_repeated, NdIndexOffsetHelper<int64_t, 3>& input_helper,
                  const int64_t max_input_length, const int64_t batch_size,
                  const int64_t num_labels) {
    FOR_RANGE(int64_t, b, 0, batch_size) {
      int64_t input_length = input_lengths_ptr[b];
      int previous_label = 0, t_dec = 0;
      CHECK_GE(max_input_length, input_length);
      for (int64_t t = 0; t < input_length; ++t) {
        const T* prob_data = &log_probs_ptr[input_helper.NdIndexToOffset(t, b, 0)];
        int curr_label = std::max_element(prob_data, prob_data + num_labels) - prob_data;
        if (curr_label != 0 && (!merge_repeated || (previous_label != curr_label))) {
          t_dec++;
          decoded_ptr[b * max_input_length + t_dec] = curr_label;
        }
        previous_label = curr_label;
      }
      for (int64_t t = t_dec + 1; t < max_input_length; ++t) {
        decoded_ptr[b * max_input_length + t] = 0;
      }
    }
  }
};

}  // namespace

REGISTER_CTC_GREEDY_DECODER_KERNELS(DeviceType::kCPU, float);
REGISTER_CTC_GREEDY_DECODER_KERNELS(DeviceType::kCPU, double);

}  // namespace oneflow
