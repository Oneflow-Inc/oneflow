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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_ND_INDEX_OFFSET_CALCULATOR_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_ND_INDEX_OFFSET_CALCULATOR_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/primitive/fast_math.h"
#include <cassert>

namespace oneflow {

template<typename T, int N>
class NdIndexOffsetHelperCalculator {
 public:
  NdIndexOffsetHelperCalculator() {}
  template<class... Ts>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelperCalculator(T d0, Ts... dims) {
    constexpr int n = 1 + sizeof...(dims);
    static_assert(n <= N, "");
    T dims_arr[n] = {d0, static_cast<T>(dims)...};
    InitFastIntegerMath(dims_arr, n);
  }

  OF_DEVICE_FUNC explicit NdIndexOffsetHelperCalculator(const T* dims) { InitFastIntegerMath(dims, N); }

  template<typename U>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelperCalculator(const U* dims) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) { dims_arr[i] = dims[i]; }
    InitFastIntegerMath(dims_arr, N);
  }

  OF_DEVICE_FUNC explicit NdIndexOffsetHelperCalculator(const T* dims, int n) {
    InitFastIntegerMath(dims, n);
  }

  template<typename U>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelperCalculator(const U* dims, int n) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      if (i < n) { dims_arr[i] = dims[i]; }
    }
    InitFastIntegerMath(dims_arr, n);
  }

  ~NdIndexOffsetHelperCalculator() = default;

  OF_DEVICE_FUNC T NdIndexToOffset(const T* index) const {
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) { offset += math_helper_[i].mul(index[i]); }
    offset += index[N - 1];
    return offset;
  }

  OF_DEVICE_FUNC T NdIndexToOffset(const T* index, int n) const {
    assert(n <= N);
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) { offset += math_helper_[i].mul(index[i]); }
    }
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
    for (int i = 0; i < n - 1; ++i) { offset += math_helper_[i].mul(index[i]); }
    if (n == N) {
      offset += index[n - 1];
    } else {
      offset += math_helper_[n - 1].mul(index[n - 1]);
    }
    return offset;
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index) const {
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) {
      const T idx = math_helper_[i].divides(remaining);
      index[i] = idx;
      remaining = remaining - math_helper_[i].mul(idx);
    }
    index[N - 1] = remaining;
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index, int n) const {
    assert(n <= N);
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        const T idx = math_helper_[i].divides(remaining);
        index[i] = idx;
        remaining = remaining - math_helper_[i].mul(idx);
      }
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
      const T idx = math_helper_[i].divides(remaining);
      *index[i] = idx;
      remaining = remaining - math_helper_[i].mul(idx);
    }
    if (n == N) {
      *index[n - 1] = remaining;
    } else {
      *index[n - 1] = math_helper_[n - 1].divides(remaining);
    }
  }

  OF_DEVICE_FUNC constexpr int Size() const { return N; }

 private:
  OF_DEVICE_FUNC void InitFastIntegerMath(const T* dims, const int n) {
    T stride_arr[N];
    for (int i = n - 1; i < N; ++i) {
      stride_arr[i] = 1;
      math_helper_[i] = FastIntegerMath<T>(1);
    }
    for (int i = n - 2; i >= 0; --i) {
      stride_arr[i] = dims[i + 1] * stride_arr[i + 1];
      math_helper_[i] = FastIntegerMath<T>(stride_arr[i]);
    }
  }
  FastIntegerMath<T> math_helper_[N];
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_ND_INDEX_OFFSET_CALCULATOR_H_
