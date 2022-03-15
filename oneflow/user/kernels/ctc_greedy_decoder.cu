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
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace {

template<typename T>
__global__ void CtcGreedyDecodeGpuMultiThread(int64_t* decoded_ptr, T* neg_sum_logits_ptr,
                                              const T* log_probs_ptr,
                                              const int64_t* input_lengths_ptr,
                                              const bool merge_repeated,
                                              const int64_t max_input_length,
                                              const int64_t batch_size, const int64_t num_labels) {
  const int64_t bid = blockIdx.x;
  const int64_t tid = threadIdx.x;

  for (int64_t b = bid; b < batch_size; b += gridDim.x) {
    if (tid == 0) {
      if (input_lengths_ptr[b] > max_input_length) __trap();
    }
  }

  for (int64_t b = bid; b < batch_size; b += gridDim.x) {
    extern __shared__ int64_t shared_max_indices_memory[];
    int64_t* shared_max_indices = (int64_t*)shared_max_indices_memory;
    NdIndexOffsetHelper<int64_t, 3> input_helper(max_input_length, batch_size, num_labels);
    for (int64_t t = tid; t < max_input_length; t += blockDim.x) {
      const T* prob_data_t = &log_probs_ptr[input_helper.NdIndexToOffset(t, b, 0)];
      int64_t max_indice = 0;
      T max_value = -FLT_MAX;
      FOR_RANGE(int64_t, c, 0, num_labels) {
        const T prob = prob_data_t[c];
        if (prob > max_value) {
          max_indice = c;
          max_value = prob;
        }
      }
      shared_max_indices[t] = max_indice;
    }

    __syncthreads();

    if (tid == 0) {
      int64_t prev_indices = -1, t_dec = 0;
      FOR_RANGE(int64_t, t, 0, input_lengths_ptr[b]) {
        const T* prob_data_t = &log_probs_ptr[input_helper.NdIndexToOffset(t, b, 0)];
        const int64_t indice_t = shared_max_indices[t];
        neg_sum_logits_ptr[b] -= prob_data_t[indice_t];
        if (indice_t != num_labels - 1 && !(merge_repeated && (prev_indices == indice_t))) {
          decoded_ptr[b * max_input_length + t_dec] = indice_t;
          t_dec++;
        }
        prev_indices = indice_t;
      }
      FOR_RANGE(int64_t, t, t_dec, max_input_length) { decoded_ptr[b * max_input_length + t] = 0; }
    }
  }
}

template<typename T>
__global__ void CtcGreedyDecodeGpu(int64_t* decoded_ptr, T* neg_sum_logits_ptr,
                                   const T* log_probs_ptr, const int64_t* input_lengths_ptr,
                                   const bool merge_repeated, const int64_t max_input_length,
                                   const int64_t batch_size, const int64_t num_labels) {
  for (int64_t b = 0; b < batch_size; b++) {
    if (input_lengths_ptr[b] > max_input_length) __trap();
  }
  NdIndexOffsetHelper<int64_t, 3> input_helper(max_input_length, batch_size, num_labels);

  CUDA_1D_KERNEL_LOOP(b, batch_size) {
    int prev_indices = -1, t_dec = 0;
    neg_sum_logits_ptr[b] = 0;
    FOR_RANGE(int64_t, t, 0, input_lengths_ptr[b]) {
      const T* prob_data_t = &log_probs_ptr[input_helper.NdIndexToOffset(t, b, 0)];
      int64_t max_indice = -1;
      T max_value = -FLT_MAX;
      FOR_RANGE(int64_t, c, 0, num_labels) {
        if (prob_data_t[c] > max_value) {
          max_indice = c;
          max_value = prob_data_t[c];
        }
      }
      neg_sum_logits_ptr[b] -= max_value;
      if (max_indice != num_labels - 1 && !(merge_repeated && (prev_indices == max_indice))) {
        decoded_ptr[b * max_input_length + t_dec] = max_indice;
        t_dec++;
      }
      prev_indices = max_indice;
    }
    FOR_RANGE(int64_t, t, t_dec, max_input_length) { decoded_ptr[b * max_input_length + t] = 0; }
  }
}

template<typename T>
struct CTCGreedyDecoderFunctor<DeviceType::kCUDA, T> final {
  void operator()(ep::Stream* stream, int64_t* decoded_ptr, T* neg_sum_logits_ptr,
                  const T* log_probs_ptr, const int64_t* input_lengths_ptr,
                  const bool merge_repeated, const int64_t max_input_length,
                  const int64_t batch_size, const int64_t num_labels) {
    int32_t thread_num = batch_size * kCudaThreadsNumPerBlock;
    int64_t shared_mem_size = max_input_length * sizeof(int64_t);

    int max_active_blocks;
    OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, CtcGreedyDecodeGpu<T>, kCudaThreadsNumPerBlock, shared_mem_size));
    if (max_active_blocks > 0) {
      CtcGreedyDecodeGpuMultiThread<<<BlocksNum4ThreadsNum(thread_num), kCudaThreadsNumPerBlock,
                                      shared_mem_size,
                                      stream->As<ep::CudaStream>()->cuda_stream()>>>(
          decoded_ptr, neg_sum_logits_ptr, log_probs_ptr, input_lengths_ptr, merge_repeated,
          max_input_length, batch_size, num_labels);

    } else {
      CtcGreedyDecodeGpu<<<BlocksNum4ThreadsNum(thread_num), kCudaThreadsNumPerBlock, 0,
                           stream->As<ep::CudaStream>()->cuda_stream()>>>(
          decoded_ptr, neg_sum_logits_ptr, log_probs_ptr, input_lengths_ptr, merge_repeated,
          max_input_length, batch_size, num_labels);
    }
  }
};

}  // namespace

REGISTER_CTC_GREEDY_DECODER_KERNELS(DeviceType::kCUDA, float);
REGISTER_CTC_GREEDY_DECODER_KERNELS(DeviceType::kCUDA, double);

}  // namespace oneflow
