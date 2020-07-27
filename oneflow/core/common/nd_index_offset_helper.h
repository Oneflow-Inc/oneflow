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
#ifndef ONEFLOW_CORE_COMMON_ND_INDEX_OFFSET_HELPER_H_
#define ONEFLOW_CORE_COMMON_ND_INDEX_OFFSET_HELPER_H_

#include "oneflow/core/common/data_type.h"
#include <cassert>

namespace oneflow {

template<typename T, int N>
class NdIndexOffsetHelper {
 public:
  template<class... Ts>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(T d0, Ts... dims) {
    constexpr int n = 1 + sizeof...(dims);
    static_assert(n <= N, "");
    T dims_arr[n] = {d0, static_cast<T>(dims)...};
    InitStrides(dims_arr, n);
  }

  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims) { InitStrides(dims, N); }

  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims, int n) { InitStrides(dims, n); }

  OF_DEVICE_FUNC ~NdIndexOffsetHelper() = default;

  OF_DEVICE_FUNC T NdIndexToOffset(const T* index) const {
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) { offset += index[i] * stride_[i]; }
    offset += index[N - 1];
    return offset;
  }

  OF_DEVICE_FUNC T NdIndexToOffset(const T* index, int n) const {
    assert(n <= N);
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n; ++i) { offset += index[i] * stride_[i]; }
    return offset;
  }

  template<class... Ts>
  OF_DEVICE_FUNC T NdIndexToOffset(T d0, Ts... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T index[n] = {d0, others...};
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) { offset += index[i] * stride_[i]; }
    if (n == N) {
      offset += index[n - 1];
    } else {
      offset += index[n - 1] * stride_[n - 1];
    }
    return offset;
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index) const {
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) {
      const T idx = remaining / stride_[i];
      index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    index[N - 1] = remaining;
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index, int n) const {
    assert(n <= N);
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n; ++i) {
      const T idx = remaining / stride_[i];
      index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
  }

  template<class... Ts>
  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T& d0, Ts&... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T* index[n] = {&d0, &others...};
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) {
      const T idx = remaining / stride_[i];
      *index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    if (n == N) {
      *index[n - 1] = remaining;
    } else {
      *index[n - 1] = remaining / stride_[n - 1];
    }
  }

  OF_DEVICE_FUNC constexpr int Size() const { return N; }

 private:
  OF_DEVICE_FUNC void InitStrides(const T* dims, const int n) {
    for (int i = n - 1; i < N; ++i) { stride_[i] = 1; }
    for (int i = n - 2; i >= 0; --i) { stride_[i] = dims[i + 1] * stride_[i + 1]; }
  }

  T stride_[N];
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ND_INDEX_OFFSET_HELPER_H_
