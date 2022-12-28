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

#include "oneflow/user/kernels/embedding_kernel_util.h"

namespace oneflow {

template<typename T, typename IndexType>
struct EmbeddingReNormFunctor<DeviceType::kCPU, T, IndexType> final {
  void operator()(ep::Stream* stream, const T* in_buf, const IndexType* indices_buf, T* out_buf,
                  const double max_norm, const double norm_type, const int64_t num_indices,
                  const int64_t emb_size, const int64_t emb_dim, int32_t* tmp_buf) {
    auto sorted_indices = std::vector<IndexType>(indices_buf, indices_buf + num_indices);
    std::sort(sorted_indices.begin(), sorted_indices.end());

    for (int64_t i = 0; i < num_indices; i++) {
      if (i > 0 && sorted_indices[i] == sorted_indices[i - 1]) { continue; }
      CHECK(sorted_indices[i] >= 0 && sorted_indices[i] < emb_size);
      double norm = 0;
      for (int64_t j = emb_dim * sorted_indices[i]; j < emb_dim * (sorted_indices[i] + 1); j++) {
        norm += std::pow(std::abs(in_buf[j]), norm_type);
      }
      norm = std::pow(norm, (1.0 / norm_type));
      if (norm > max_norm) {
        double scale = max_norm / (norm + 1e-7);
        for (int64_t j = emb_dim * sorted_indices[i]; j < emb_dim * (sorted_indices[i] + 1); j++) {
          out_buf[j] = in_buf[j] * scale;
        }
      }
    }
  }
};

template<typename T, typename IndexType>
struct EmbeddingFunctor<DeviceType::kCPU, T, IndexType> final {
  void operator()(ep::Stream* stream, const T* weight_buf, const IndexType* indices_buf, T* out_buf,
                  const int64_t padding_idx, const bool scale_grad_by_freq,
                  const int64_t num_indices, const int64_t emb_size, const int64_t emb_dim) {
    for (int64_t i = 0; i < num_indices; i++) {
      IndexType indice = indices_buf[i];
      CHECK(indice >= 0 && indice < emb_size);
      const T* from = weight_buf + indice * emb_dim;
      T* to = out_buf + i * emb_dim;
      std::copy(from, from + emb_dim, to);
    }
  }
};

template<typename T, typename IndexType>
struct EmbeddingGradFunctor<DeviceType::kCPU, T, IndexType> final {
  void operator()(ep::Stream* stream, const T* dy_buf, const IndexType* indices_buf, T* dx_buf,
                  const int64_t padding_idx, const bool scale_grad_by_freq,
                  const int64_t num_indices, const int64_t emb_size, const int64_t emb_dim,
                  int32_t* tmp_buf) {
    for (int64_t i = 0; i < num_indices; i++) {
      IndexType indice = indices_buf[i];
      if (indice != padding_idx) {
        const T* from = dy_buf + i * emb_dim;
        T* to = dx_buf + indice * emb_dim;
        std::transform(from, from + emb_dim, to, to, std::plus<T>());
      }
    }

    if (scale_grad_by_freq) {
      std::vector<IndexType> indice_freq(emb_size, 0);
      for (int64_t i = 0; i < num_indices; i++) { indice_freq[indices_buf[i]]++; }

      for (int64_t i = 0; i < emb_size; i++) {
        if (indice_freq[i] > 1) {
          T* from = dx_buf + i * emb_dim;
          for (int64_t j = 0; j < emb_dim; j++) { from[j] /= indice_freq[i]; }
        }
      }
    }
  }
};

#define INITIATE_EMBEDDING_KERNEL_UTIL_CPU_IMPL(in_type_pair, index_type_pair)             \
  template struct EmbeddingReNormFunctor<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                         OF_PP_PAIR_FIRST(index_type_pair)>;               \
  template struct EmbeddingFunctor<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair),       \
                                   OF_PP_PAIR_FIRST(index_type_pair)>;                     \
  template struct EmbeddingGradFunctor<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair),   \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_EMBEDDING_KERNEL_UTIL_CPU_IMPL,
                                 EMBEDDING_DATA_TYPE_SEQ_CPU, INDEX_DATA_TYPE_SEQ);
#undef INITIATE_EMBEDDING_KERNEL_UTIL_CPU_IMPL

}  // namespace oneflow
