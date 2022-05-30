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
struct AccumulateType {
  using type = T;
};

template<>
struct AccumulateType<half> {
  using type = float;
};

template<typename T, typename IndexType>
__global__ void embedding_kernel(const T* weight_buf, const IndexType* indices_buf, T* out_buf,
                                 const int64_t num_indices, const int64_t emb_size,
                                 const int64_t emb_dim) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, num_indices * emb_dim) {
    IndexType indices_index = i / emb_dim;
    IndexType emb_dim_index = i - indices_index * emb_dim;
    IndexType emb_size_index = indices_buf[indices_index];
    assert(emb_size_index >= 0 && emb_size_index < emb_size);
    IndexType from_index = emb_size_index * emb_dim + emb_dim_index;
    out_buf[i] = weight_buf[from_index];
  }
}

template<typename T, typename IndexType>
__global__ void embedding_grad_kernel(const T* dy_buf, const IndexType* indices_buf, T* dx_buf,
                                      const int64_t padding_idx, const int64_t num_indices,
                                      const int64_t emb_dim) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, num_indices * emb_dim) {
    IndexType indices_index = i / emb_dim;
    IndexType emb_dim_index = i - indices_index * emb_dim;
    IndexType emb_size_index = indices_buf[indices_index];
    if (emb_size_index != padding_idx) {
      IndexType from_index = emb_size_index * emb_dim + emb_dim_index;
      cuda::atomic::Add(dx_buf + from_index, dy_buf[i]);
    }
  }
}

template<typename IndexType>
__global__ void indices_freq_kernel(const IndexType* indices_buf, const int64_t num_indices,
                                    int32_t* indices_freq, const int64_t emb_size) {
  CUDA_1D_KERNEL_LOOP_T(IndexType, i, num_indices) {
    IndexType index = indices_buf[i];
    assert(index >= 0 && index < emb_size);
    cuda::atomic::Add(indices_freq + index, 1);
  }
}

template<typename T, typename IndexType>
__global__ void emb_scale_kernel(T* dx_buf, const int64_t emb_size, const int64_t emb_dim,
                                 int32_t* indices_freq) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, emb_size * emb_dim) {
    IndexType emb_size_index = i / emb_dim;
    if (indices_freq[emb_size_index] > 1) {
      dx_buf[i] /= static_cast<T>(indices_freq[emb_size_index]);
    }
  }
}

template<typename T, typename IndexType, typename AccumType>
__global__ void embedding_renorm_kernel(const T* in_buf, T* out_buf, int32_t* indices_freq,
                                        const AccumType max_norm, const AccumType norm_type,
                                        const int64_t emb_size, const int64_t emb_dim) {
  int64_t tid = threadIdx.x;
  for (int64_t emb_idx = blockIdx.x; emb_idx < emb_size; emb_idx += gridDim.x) {
    if (indices_freq[emb_idx] == 0) { continue; }
    int64_t base_index = emb_idx * emb_dim;

    AccumType v = 0;
    for (int64_t i = tid; i < emb_dim; i += blockDim.x) {
      v += pow(abs(static_cast<AccumType>(in_buf[base_index + i])), norm_type);
    }

    using BlockReduce = cub::BlockReduce<AccumType, kCudaThreadsNumPerBlock>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ AccumType norm;
    v = BlockReduce(temp_storage).Sum(v);

    if (tid == 0) { norm = pow(v, static_cast<AccumType>(1.0 / norm_type)); }
    __syncthreads();

    if (norm > max_norm) {
      auto scale = static_cast<T>(max_norm / (norm + 1e-7));
      for (int64_t i = tid; i < emb_dim; i += blockDim.x) {
        out_buf[base_index + i] = in_buf[base_index + i] * scale;
      }
    }
  }
}

}  // namespace

template<typename T, typename IndexType>
struct EmbeddingReNormFunctor<DeviceType::kCUDA, T, IndexType> final {
  void operator()(ep::Stream* stream, const T* in_buf, const IndexType* indices_buf, T* out_buf,
                  const double max_norm, const double norm_type, const int64_t num_indices,
                  const int64_t emb_size, const int64_t emb_dim, int32_t* tmp_buf) {
    indices_freq_kernel<IndexType><<<BlocksNum4ThreadsNum(num_indices), kCudaThreadsNumPerBlock, 0,
                                     stream->As<ep::CudaStream>()->cuda_stream()>>>(
        indices_buf, num_indices, tmp_buf, emb_size);

    using AccumType = typename AccumulateType<T>::type;
    embedding_renorm_kernel<T, IndexType, AccumType>
        <<<BlocksNum4ThreadsNum(emb_size), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(
            in_buf, out_buf, tmp_buf, static_cast<AccumType>(max_norm),
            static_cast<AccumType>(norm_type), emb_size, emb_dim);
  }
};

template<typename T, typename IndexType>
struct EmbeddingFunctor<DeviceType::kCUDA, T, IndexType> final {
  void operator()(ep::Stream* stream, const T* weight_buf, const IndexType* indices_buf, T* out_buf,
                  const int64_t padding_idx, const bool scale_grad_by_freq,
                  const int64_t num_indices, const int64_t emb_size, const int64_t emb_dim) {
    embedding_kernel<T, IndexType>
        <<<BlocksNum4ThreadsNum(num_indices * emb_dim), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(weight_buf, indices_buf, out_buf,
                                                          num_indices, emb_size, emb_dim);
  }
};

template<typename T, typename IndexType>
struct EmbeddingGradFunctor<DeviceType::kCUDA, T, IndexType> final {
  void operator()(ep::Stream* stream, const T* dy_buf, const IndexType* indices_buf, T* dx_buf,
                  const int64_t padding_idx, const bool scale_grad_by_freq,
                  const int64_t num_indices, const int64_t emb_size, const int64_t emb_dim,
                  int32_t* tmp_buf) {
    embedding_grad_kernel<T, IndexType>
        <<<BlocksNum4ThreadsNum(num_indices * emb_dim), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(dy_buf, indices_buf, dx_buf, padding_idx,
                                                          num_indices, emb_dim);
    if (scale_grad_by_freq) {
      indices_freq_kernel<IndexType><<<BlocksNum4ThreadsNum(num_indices), kCudaThreadsNumPerBlock,
                                       0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          indices_buf, num_indices, tmp_buf, emb_size);
      emb_scale_kernel<T, IndexType>
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
