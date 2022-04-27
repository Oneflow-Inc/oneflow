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

#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/user/kernels/embedding_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T, typename index_T>
__global__ void embedding_kernel(const T* weight_buf, const index_T* indices_buf, T* out_buf,
                                 const int32_t num_indices, const int32_t emb_dim) {
  CUDA_1D_KERNEL_LOOP(i, num_indices * emb_dim) {
    int32_t indices_index = i / emb_dim;
    int32_t emb_dim_index = i - indices_index * emb_dim;
    int32_t from_index = indices_buf[indices_index] * emb_dim + emb_dim_index;
    out_buf[i] = weight_buf[from_index];
  }
}

template<typename T, typename index_T>
__global__ void embedding_grad_kernel(const T* dy_buf, const index_T* indices_buf, T* dx_buf,
                                      const int32_t padding_idx, const int32_t num_indices,
                                      const int32_t emb_dim) {
  CUDA_1D_KERNEL_LOOP(i, num_indices * emb_dim) {
    int32_t indices_index = i / emb_dim;
    int32_t emb_dim_index = i - indices_index * emb_dim;
    int32_t emb_size_index = indices_buf[indices_index];
    int32_t from_index = emb_size_index * emb_dim + emb_dim_index;
    if (emb_size_index != padding_idx) { cuda::atomic::Add(dx_buf + from_index, dy_buf[i]); }
  }
}

template<typename index_T>
__global__ void indicesFreq(const index_T* indices_buf, const int32_t num_indices,
                            int32_t* tmp_buf) {
  CUDA_1D_KERNEL_LOOP(i, num_indices) { cuda::atomic::Add(tmp_buf + indices_buf[i], 1); }
}

template<typename T, typename index_T>
__global__ void embeddingScale(T* dx_buf, const int32_t emb_size, const int32_t emb_dim,
                               int32_t* tmp_buf) {
  CUDA_1D_KERNEL_LOOP(i, emb_size * emb_dim) {
    int32_t emb_size_index = i / emb_dim;
    if (tmp_buf[emb_size_index] > 1) { dx_buf[i] /= tmp_buf[emb_size_index]; }
  }
}

template<typename T, typename index_T>
__global__ void embNorm(const T* in_buf, int32_t* indices_frep, double* emb_norm,
                        const double norm_type, const int32_t emb_size, const int32_t emb_dim) {
  CUDA_1D_KERNEL_LOOP(i, emb_size * emb_dim) {
    int32_t emb_size_index = i / emb_dim;
    int32_t emb_dim_index = i - emb_size_index * emb_dim;

    if (indices_frep[emb_size_index] > 0) {
      double v;
      double item = static_cast<double>(in_buf[i]);
      if (norm_type == 1) {
        v = std::abs(item);
      } else if (norm_type == 2) {
        v = item * item;
      } else {
        v = std::pow(item, norm_type);
      }
      cuda::atomic::Add(emb_norm + emb_size_index, v);
    }
  }
}

template<typename T, typename index_T>
__global__ void embNorm_kernel(const T* in_buf, T* out_buf, double* emb_norm, int32_t* indices_frep,
                               const double max_norm, const int32_t emb_size,
                               const int32_t emb_dim) {
  CUDA_1D_KERNEL_LOOP(i, emb_size * emb_dim) {
    int32_t emb_size_index = i / emb_dim;
    int32_t emb_dim_index = i - emb_size_index * emb_dim;
    if (indices_frep[emb_size_index] > 0) {
      double v = emb_norm[emb_size_index];
      if (v > max_norm) {
        T scale = static_cast<T>(max_norm / (v + 1e-7));
        out_buf[i] = in_buf[i] * scale;
      }
    }
  }
}

}  // namespace

template<typename T, typename index_T>
struct EmbeddingRenormFunctor<DeviceType::kCUDA, T, index_T> final {
  void operator()(ep::Stream* stream, const T* in_buf, const index_T* indices_buf, T* out_buf,
                  const double max_norm, const double norm_type, const int32_t num_indices,
                  const int32_t emb_size, const int32_t emb_dim, void* tmp_buf) {
    size_t bytes_used = 0;
    int32_t* indices_frep = reinterpret_cast<int32_t*>(static_cast<char*>(tmp_buf) + bytes_used);
    bytes_used += sizeof(int32_t) * emb_size * 2;
    double* emb_norm = reinterpret_cast<double*>(static_cast<char*>(tmp_buf) + bytes_used);
    bytes_used += sizeof(double) * emb_size;

    indicesFreq<index_T>
        <<<BlocksNum4ThreadsNum(num_indices), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(indices_buf, num_indices, indices_frep);
    embNorm<T, index_T><<<BlocksNum4ThreadsNum(emb_size * emb_dim), kCudaThreadsNumPerBlock, 0,
                          stream->As<ep::CudaStream>()->cuda_stream()>>>(
        in_buf, indices_frep, emb_norm, norm_type, emb_size, emb_dim);
    embNorm_kernel<T, index_T><<<BlocksNum4ThreadsNum(emb_size * emb_dim), kCudaThreadsNumPerBlock,
                                 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        in_buf, out_buf, emb_norm, indices_frep, max_norm, emb_size, emb_dim);
  }
};

template<typename T, typename index_T>
struct EmbeddingFunctor<DeviceType::kCUDA, T, index_T> final {
  void operator()(ep::Stream* stream, const T* weight_buf, const index_T* indices_buf, T* out_buf,
                  const int32_t padding_idx, const bool scale_grad_by_freq,
                  const int32_t num_indices, const int32_t emb_size, const int32_t emb_dim) {
    embedding_kernel<T, index_T>
        <<<BlocksNum4ThreadsNum(num_indices * emb_dim), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(weight_buf, indices_buf, out_buf,
                                                          num_indices, emb_dim);
  }
};

template<typename T, typename index_T>
struct EmbeddingGradFunctor<DeviceType::kCUDA, T, index_T> final {
  void operator()(ep::Stream* stream, const T* dy_buf, const index_T* indices_buf, T* dx_buf,
                  const int32_t padding_idx, const bool scale_grad_by_freq,
                  const int32_t num_indices, const int32_t emb_size, const int32_t emb_dim,
                  int32_t* tmp_buf) {
    embedding_grad_kernel<T, index_T>
        <<<BlocksNum4ThreadsNum(num_indices * emb_dim), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(dy_buf, indices_buf, dx_buf, padding_idx,
                                                          num_indices, emb_dim);
    if (scale_grad_by_freq) {
      indicesFreq<index_T>
          <<<BlocksNum4ThreadsNum(num_indices), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(indices_buf, num_indices, tmp_buf);
      embeddingScale<T, index_T>
          <<<BlocksNum4ThreadsNum(emb_size * emb_dim), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(dx_buf, emb_size, emb_dim, tmp_buf);
    }
  }
};

#define INITIATE_EMBEDDING_KERNEL_UTIL_CUDA_IMPL(in_type_pair, index_type_pair)             \
  template struct EmbeddingRenormFunctor<DeviceType::kCUDA, OF_PP_PAIR_FIRST(in_type_pair), \
                                         OF_PP_PAIR_FIRST(index_type_pair)>;                \
  template struct EmbeddingFunctor<DeviceType::kCUDA, OF_PP_PAIR_FIRST(in_type_pair),       \
                                   OF_PP_PAIR_FIRST(index_type_pair)>;                      \
  template struct EmbeddingGradFunctor<DeviceType::kCUDA, OF_PP_PAIR_FIRST(in_type_pair),   \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_EMBEDDING_KERNEL_UTIL_CUDA_IMPL,
                                 EMBEDDING_DATA_TYPE_SEQ_CUDA, INDEX_DATA_TYPE_SEQ);

#undef INITIATE_EMBEDDING_KERNEL_UTIL_CUDA_IMPL

}  // namespace oneflow
