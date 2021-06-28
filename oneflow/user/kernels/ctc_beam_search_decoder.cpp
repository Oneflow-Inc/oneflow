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
#include "oneflow/user/kernels/ctc_beam_search_decoder.h"

namespace oneflow {
namespace {

template<typename T>
struct CTCBeamSearchDecoderFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, int64_t* decoded_ptr, T* log_probability_ptr,
                  const T* log_probs_ptr, const int64_t* input_lengths_ptr,
                  const int32_t beam_width, const int32_t top_paths, const int64_t max_input_length,
                  const int64_t batch_size, const int64_t num_labels) {
    FOR_RANGE(int64_t, b, 0, batch_size) { CHECK_GE(max_input_length, input_lengths_ptr[b]); }
    NdIndexOffsetHelper<int64_t, 3> input_helper(max_input_length, batch_size, num_labels);
    // TODO
  }
};

}  // namespace

REGISTER_CTC_BEAM_SEARCH_DECODER_KERNELS(DeviceType::kCPU, float);
REGISTER_CTC_BEAM_SEARCH_DECODER_KERNELS(DeviceType::kCPU, double);

}  // namespace oneflow
