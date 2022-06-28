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

#ifndef ONEFLOW_USER_KERNELS_EMBEDDING_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_EMBEDDING_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename IndexType>
struct EmbeddingReNormFunctor final {
  void operator()(ep::Stream* stream, const T* in_buf, const IndexType* indices_buf, T* out_buf,
                  const double max_norm, const double norm_type, const int64_t num_indices,
                  const int64_t emb_size, const int64_t emb_dim, int32_t* tmp_buf);
};

template<DeviceType device_type, typename T, typename IndexType>
struct EmbeddingFunctor final {
  void operator()(ep::Stream* stream, const T* weight_buf, const IndexType* indices_buf, T* out_buf,
                  const int64_t padding_idx, const bool scale_grad_by_freq,
                  const int64_t num_indices, const int64_t emb_size, const int64_t emb_dim);
};

template<DeviceType device_type, typename T, typename IndexType>
struct EmbeddingGradFunctor final {
  void operator()(ep::Stream* stream, const T* dy_buf, const IndexType* indices_buf, T* dx_buf,
                  const int64_t padding_idx, const bool scale_grad_by_freq,
                  const int64_t num_indices, const int64_t emb_size, const int64_t emb_dim,
                  int32_t* tmp_buf);
};

#define EMBEDDING_DATA_TYPE_SEQ_CPU FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
#define EMBEDDING_DATA_TYPE_SEQ_CUDA FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_EMBEDDING_KERNEL_UTIL_H_
