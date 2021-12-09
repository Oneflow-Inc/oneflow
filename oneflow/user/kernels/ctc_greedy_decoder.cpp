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
  void operator()(ep::Stream* stream, int64_t* decoded_ptr, T* neg_sum_logits_ptr,
                  const T* log_probs_ptr, const int64_t* input_lengths_ptr,
                  const bool merge_repeated, const int64_t max_input_length,
                  const int64_t batch_size, const int64_t num_labels) {
    FOR_RANGE(int64_t, b, 0, batch_size) { CHECK_GE(max_input_length, input_lengths_ptr[b]); }
    NdIndexOffsetHelper<int64_t, 3> input_helper(max_input_length, batch_size, num_labels);

    FOR_RANGE(int64_t, b, 0, batch_size) {
      int64_t prev_indices = -1, t_dec = 0;
      neg_sum_logits_ptr[b] = 0;
      FOR_RANGE(int64_t, t, 0, input_lengths_ptr[b]) {
        const T* prob_data_t = &log_probs_ptr[input_helper.NdIndexToOffset(t, b, 0)];
        int64_t max_indice = std::max_element(prob_data_t, prob_data_t + num_labels) - prob_data_t;
        neg_sum_logits_ptr[b] -= prob_data_t[max_indice];
        if (max_indice != num_labels - 1 && !(merge_repeated && (prev_indices == max_indice))) {
          decoded_ptr[b * max_input_length + t_dec] = max_indice;
          t_dec++;
        }
        prev_indices = max_indice;
      }
      FOR_RANGE(int64_t, t, t_dec, max_input_length) { decoded_ptr[b * max_input_length + t] = 0; }
    }
  }
};

}  // namespace

REGISTER_CTC_GREEDY_DECODER_KERNELS(DeviceType::kCPU, float);
REGISTER_CTC_GREEDY_DECODER_KERNELS(DeviceType::kCPU, double);

}  // namespace oneflow
