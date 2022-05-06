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

#include <cub/cub.cuh>
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/user/kernels/embedding_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
struct Abs {
  __device__ __forceinline__ T operator()(T x) { return abs(x); }
};

template<>
struct Abs<half> {
  __device__ __forceinline__ half operator()(half x) { return __habs(x); }
};

template<typename T>
struct Pow {
  __device__ __forceinline__ T operator()(T x, T base) { return pow(x, base); }
};

template<>
struct Pow<half> {
  __device__ __forceinline__ half operator()(half x, half base) {
    return static_cast<half>(pow(static_cast<float>(x), static_cast<float>(base)));
  }
};

template<typename T, typename index_T>
__global__ void embedding_kernel(const T* weight_buf, const index_T* indices_buf, T* out_buf,
                                 const int32_t num_indices, const int32_t emb_size,
                                 const int32_t emb_dim) {
  CUDA_1D_KERNEL_LOOP(i, num_indices * emb_dim) {
    int32_t indices_index = i / emb_dim;
    int32_t emb_dim_index = i - indices_index * emb_dim;
    int32_t emb_size_index = indices_buf[indices_index];
    assert(emb_size_index >= 0 && emb_size_index < emb_size);
    int32_t from_index = emb_size_index * emb_dim + emb_dim_index;
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
    if (emb_size_index != padding_idx) {
      int32_t from_index = emb_size_index * emb_dim + emb_dim_index;
      cuda::atomic::Add(dx_buf + from_index, dy_buf[i]);
    }
  }
}

template<typename index_T>
__global__ void renorm_indices_freq_kernel(const index_T* indices_buf, const int32_t num_indices,
                                           int32_t* indices_frep, const int32_t emb_size) {
  CUDA_1D_KERNEL_LOOP(i, num_indices) {
    int32_t index = indices_buf[i];
    assert(index >= 0 && index < emb_size);
    cuda::atomic::Add(indices_frep + index, 1);
  }
}

template<typename index_T>
__global__ void grad_indices_freq_kernel(const index_T* indices_buf, const int32_t num_indices,
                                         int32_t* indices_frep) {
  CUDA_1D_KERNEL_LOOP(i, num_indices) { cuda::atomic::Add(indices_frep + indices_buf[i], 1); }
}

template<typename T, typename index_T>
__global__ void emb_scale_kernel(T* dx_buf, const int32_t emb_size, const int32_t emb_dim,
                                 int32_t* tmp_buf) {
  CUDA_1D_KERNEL_LOOP(i, emb_size * emb_dim) {
    int32_t emb_size_index = i / emb_dim;
    if (tmp_buf[emb_size_index] > 1) { dx_buf[i] /= tmp_buf[emb_size_index]; }
  }
}

template<typename T, typename index_T>
__global__ void embedding_renorm_kernel(const T* in_buf, T* out_buf, int32_t* indices_frep,
                                        const double max_norm, const double norm_type,
                                        const int32_t emb_dim, Abs<T> abs_func, Pow<T> pow_func) {
  if (indices_frep[blockIdx.x] == 0) { return; }

  int tid = threadIdx.x;
  int base_index = blockIdx.x * emb_dim;

  T v = 0;
  for (int i = tid; i < emb_dim; i += blockDim.x) {
    v += pow_func(abs_func(in_buf[base_index + i]), static_cast<T>(norm_type));
  }

  using BlockReduce = cub::BlockReduce<T, kCudaThreadsNumPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T norm;
  v = BlockReduce(temp_storage).Sum(v);

  if (tid == 0) { norm = pow_func(v, static_cast<T>(1.0 / norm_type)); }
  __syncthreads();

  if (norm > static_cast<T>(max_norm)) {
    T scale = static_cast<T>(max_norm) / (norm + static_cast<T>(1e-7));
    for (int i = tid; i < emb_dim; i += blockDim.x) {
      out_buf[base_index + i] = in_buf[base_index + i] * scale;
    }
  }
}

}  // namespace

template<typename T, typename index_T>
struct EmbeddingReNormFunctor<DeviceType::kCUDA, T, index_T> final {
  void operator()(ep::Stream* stream, const T* in_buf, const index_T* indices_buf, T* out_buf,
                  const double max_norm, const double norm_type, const int32_t num_indices,
                  const int32_t emb_size, const int32_t emb_dim, int32_t* tmp_buf) {
    renorm_indices_freq_kernel<index_T>
        <<<BlocksNum4ThreadsNum(num_indices), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(indices_buf, num_indices, tmp_buf,
                                                          emb_size);
    Abs<T> abs_func;
    Pow<T> pow_func;
    embedding_renorm_kernel<T, index_T>
        <<<emb_size, kCudaThreadsNumPerBlock, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            in_buf, out_buf, tmp_buf, max_norm, norm_type, emb_dim, abs_func, pow_func);
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
                                                          num_indices, emb_size, emb_dim);
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
      grad_indices_freq_kernel<index_T>
          <<<BlocksNum4ThreadsNum(num_indices), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(indices_buf, num_indices, tmp_buf);
      emb_scale_kernel<T, index_T>
          <<<BlocksNum4ThreadsNum(emb_size * emb_dim), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(dx_buf, emb_size, emb_dim, tmp_buf);
    }
  }
};

#define INITIATE_EMBEDDING_KERNEL_UTIL_CUDA_IMPL(in_type_pair, index_type_pair)             \
  template struct EmbeddingReNormFunctor<DeviceType::kCUDA, OF_PP_PAIR_FIRST(in_type_pair), \
                                         OF_PP_PAIR_FIRST(index_type_pair)>;                \
  template struct EmbeddingFunctor<DeviceType::kCUDA, OF_PP_PAIR_FIRST(in_type_pair),       \
                                   OF_PP_PAIR_FIRST(index_type_pair)>;                      \
  template struct EmbeddingGradFunctor<DeviceType::kCUDA, OF_PP_PAIR_FIRST(in_type_pair),   \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_EMBEDDING_KERNEL_UTIL_CUDA_IMPL,
                                 EMBEDDING_DATA_TYPE_SEQ_CUDA, INDEX_DATA_TYPE_SEQ);

#undef INITIATE_EMBEDDING_KERNEL_UTIL_CUDA_IMPL

}  // namespace oneflow
