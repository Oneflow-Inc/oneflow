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
__global__ void vector_diagonal_kernel(T* out_buf, const T* in_buf, int32_t size, int32_t stride) {
  CUDA_1D_KERNEL_LOOP(i, size) { out_buf[i * stride] = in_buf[i]; }
}

template<typename T>
__global__ void matrix_diagonal_kernel(T* out_buf, const T* in_buf, int32_t size, int32_t stride) {
  CUDA_1D_KERNEL_LOOP(i, size) { out_buf[i] = in_buf[i * stride]; }
}

template<typename T>
struct CTCGreedyDecoderFunctor<DeviceType::kGPU, T> final {
  void operator()(DeviceCtx* ctx, int64_t* decoded_ptr, T* neg_sum_logits_ptr,
                  const T* log_probs_ptr, const int64_t* input_lengths_ptr,
                  const bool merge_repeated, NdIndexOffsetHelper<int64_t, 3>& input_helper,
                  const int64_t max_input_length, const int64_t batch_size,
                  const int64_t num_labels) {
    // TODO
  }
};

}  // namespace

REGISTER_CTC_GREEDY_DECODER_KERNELS(DeviceType::kGPU, half);
REGISTER_CTC_GREEDY_DECODER_KERNELS(DeviceType::kGPU, float);
REGISTER_CTC_GREEDY_DECODER_KERNELS(DeviceType::kGPU, double);

}  // namespace oneflow
