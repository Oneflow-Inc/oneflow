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
#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_UNARY
#define ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_UNARY

#include "oneflow/core/ep/include/primitive/broadcast_elementwise_unary.h"
#include "oneflow/core/ep/include/primitive/fast_integer_math.h"
#include "oneflow/core/ep/common/primitive/util.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace broadcast_elementwise_unary {

constexpr size_t kMaxNumDims = 8;

template<typename T, int N>
class IndexToOffsetWithStrideCalculator {
 public:
  IndexToOffsetWithStrideCalculator() {}

  OF_DEVICE_FUNC explicit IndexToOffsetWithStrideCalculator(const T* strides) {
    InitStrides(strides, N);
  }

  template<typename U>
  OF_DEVICE_FUNC explicit IndexToOffsetWithStrideCalculator(const U* strides) {
    T strides_arr[N];
    for (int i = 0; i < N; ++i) { strides_arr[i] = strides[i]; }
    InitStrides(strides_arr, N);
  }

  OF_DEVICE_FUNC explicit IndexToOffsetWithStrideCalculator(const T* strides, int n) {
    InitStrides(strides, n);
  }

  template<typename U>
  OF_DEVICE_FUNC explicit IndexToOffsetWithStrideCalculator(const U* strides, int n) {
    T strides_arr[N];
    for (int i = 0; i < N; ++i) {
      if (i < n) { strides_arr[i] = strides[i]; }
    }
    InitStrides(strides_arr, n);
  }

  ~IndexToOffsetWithStrideCalculator() = default;

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
    for (int i = 0; i < N; ++i) {
      if (i < n) { offset += index[i] * stride_[i]; }
    }
    return offset;
  }

  OF_DEVICE_FUNC constexpr int Size() const { return N; }

 private:
  OF_DEVICE_FUNC void InitStrides(const T* strides, const int n) {
    for (int i = n; i < N; ++i) { stride_[i] = 1; }
    for (int i = n - 1; i >= 0; --i) { stride_[i] = strides[i]; }
  }

  T stride_[N];
};

template<typename T, int N>
class OffsetToIndexWithStrideCalculator {
 public:
  OffsetToIndexWithStrideCalculator() {}

  OF_DEVICE_FUNC explicit OffsetToIndexWithStrideCalculator(const T* dims) {
    InitFastIntegerMath(dims, N);
  }

  template<typename U>
  OF_DEVICE_FUNC explicit OffsetToIndexWithStrideCalculator(const U* dims) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) { dims_arr[i] = dims[i]; }
    InitFastIntegerMath(dims_arr, N);
  }

  OF_DEVICE_FUNC explicit OffsetToIndexWithStrideCalculator(const T* dims, int n) {
    InitFastIntegerMath(dims, n);
  }

  template<typename U>
  OF_DEVICE_FUNC explicit OffsetToIndexWithStrideCalculator(const U* dims, int n) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      if (i < n) { dims_arr[i] = dims[i]; }
    }
    InitFastIntegerMath(dims_arr, n);
  }

  ~OffsetToIndexWithStrideCalculator() = default;

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
      if (i == n - 1) { break; }
      if (i < n - 1) {
        const T idx = math_helper_[i].divides(remaining);
        index[i] = idx;
        remaining = remaining - math_helper_[i].mul(idx);
      }
    }
    index[n - 1] = remaining;
  }

  OF_DEVICE_FUNC T divides(T remaining, int64_t i) const {
    return math_helper_[i].divides(remaining);
  }

  OF_DEVICE_FUNC T mul(T idx, int64_t i) const { return math_helper_[i].mul(idx); }

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

#define UNARY_IDENTITY_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kIdentity)
#define BROADCAST_ELEMENTWISE_CAST_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kCast)

}  // namespace broadcast_elementwise_unary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_UNARY
