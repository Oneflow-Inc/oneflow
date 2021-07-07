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

#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/user/kernels/sparse_softmax_cross_entropy_kernel_util.h"

namespace oneflow {
namespace user_op {
namespace {
template<typename T>
size_t GetReduceTempStorageSize(int64_t n, int64_t w) {
  return GetCudaAlignedSize(n * w * sizeof(T));
}
template<typename T>
size_t GetProbStorageSize(int64_t n, int64_t w) {
  return GetCudaAlignedSize(n * sizeof(T));
}
template<typename T>
size_t GetTempStorageSize(int64_t n, int64_t w) {
  return GetCudaAlignedSize(n * w * sizeof(T));
}
}  // namespace

template<typename T, typename K>
struct SparseSoftmaxCrossEntropyKernelUtil<DeviceType::kCPU, T, K> {
  static size_t GetComputeTempStorageSizeInBytes(int64_t n, int64_t w) {
    return GetReduceTempStorageSize<T>(n, w) + GetProbStorageSize<T>(n, w)
           + GetTempStorageSize<T>(n, w);
  }
  static void Compute(DeviceCtx* ctx, const int64_t n, const int64_t w, const int64_t depth,
                      const int64_t lower_bound, const T* in, T* prob, const K* labels, T* y,
                      void* temp_storage, const size_t temp_storage_bytes,
                      const MemoryCase& prob_mem_case, const MemoryCase& tmp_buffer_mem_case) {
    auto Val = NdarrayUtil<DeviceType::kCPU, T>::GetValNdarrayBuilder();
    auto Var = NdarrayUtil<DeviceType::kCPU, T>::GetVarNdarrayBuilder();
    const size_t min_temp_storage_bytes = GetComputeTempStorageSizeInBytes(n, w);
    CHECK_GE(temp_storage_bytes, min_temp_storage_bytes);
    const size_t reduce_temp_storage_bytes = GetReduceTempStorageSize<T>(n, w);
    const size_t temp_storage_bytes_offset = GetProbStorageSize<T>(n, w);
    T* reduce_storage = reinterpret_cast<T*>(temp_storage);
    auto reduce_storage_var =
        Var({static_cast<int64_t>(reduce_temp_storage_bytes / sizeof(T))}, reduce_storage);
    T* tmp = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                  + reduce_temp_storage_bytes);
    T* new_tmp = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                      + reduce_temp_storage_bytes + temp_storage_bytes_offset);

    // max | tmp[i] = Max_j(in[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceMax(ctx, Var({n, 1}, tmp), Val({n, w}, in),
                                                reduce_storage_var);
    // sub | prob[i][j] = in[i][j] - tmp[i]
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastSub(ctx, Var({n, w}, new_tmp), Val({n, w}, in),
                                                   Val({n, 1}, tmp));
    // exp | prob[i][j] = exp(prob[i][j])

    AutoMemcpy(ctx, prob, new_tmp, reduce_temp_storage_bytes, prob_mem_case, tmp_buffer_mem_case);
    NdarrayUtil<DeviceType::kCPU, T>::InplaceExp(ctx, Var({n, w}, prob));
    // sum | tmp[i] = Sum_j(prob[i][j])
    NdarrayUtil<DeviceType::kCPU, T>::ReduceSum(ctx, Var({n, 1}, tmp), Val({n, w}, prob),
                                                reduce_storage_var);
    FOR_RANGE(int64_t, i, 0, n) {
      CHECK_GE(labels[i], 0);
      CHECK_LT(labels[i], depth);
      K label = labels[i] - lower_bound;
      if (label >= 0 && label < w) { y[i] = -SafeLog(tmp[i]) + new_tmp[i * w + label]; }
    }
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t elem_cnt, const int64_t num_classes,
                          const int64_t depth, const int64_t lower_bound, const T* prob,
                          const K* labels, const T* dy, T* dx) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      const int32_t row_id = i / num_classes;
      const int32_t col_id = i - row_id * num_classes;
      CHECK_GE(labels[row_id], 0);
      CHECK_LT(labels[row_id], depth);
      K label = labels[row_id] - lower_bound;

      if (label == col_id) {
        dx[i] = dy[row_id] * (prob[i] - 1);
      } else {
        dx[i] = dy[row_id] * prob[i];
      }
    }
  }
};
#define INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CPU(data_type_pair, index_type_pair) \
  template struct SparseSoftmaxCrossEntropyKernelUtil<                                            \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_CPU
}  // namespace user_op
}  // namespace oneflow
