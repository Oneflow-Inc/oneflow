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

#define UNARY_MATH_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRelu)

#define UNARY_FLOATING_MATH_OP_SEQ                \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kElu)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kCelu)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kGelu)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kHardSwish)       \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kHardSigmoid)     \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kHardShrink)      \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kHardTanh)        \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLeakyRelu)       \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kMish)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSelu)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSilu)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSoftShrink)      \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSoftSign)        \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSoftPlus)        \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kTanh)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kThreshold)       \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAbs)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAcos)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAcosh)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAsin)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAsinh)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAtan)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAtanh)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kCeil)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kCos)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kCosh)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kErf)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kErfc)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kExp)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kExpm1)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kFloor)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLgamma)          \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLog)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLog2)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLog10)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLog1p)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLogSigmoid)      \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kNegative)        \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kReciprocal)      \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRint)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRound)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRsqrt)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSigmoid)         \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSign)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSin)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSinh)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSqrt)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSign)            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSquare)          \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kTan)             \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kTrunc)           \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kNotEqualZero)    \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kNanAssign)       \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kFastGelu)

#define UNARY_INT_MATH_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAbs)

#define UNARY_LOGICAL_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLogicalNot)

#define UNARY_UTILS_OP_SEQ              \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kIsInf) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kIsNan) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kIsFinite)

}  // namespace broadcast_elementwise_unary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_UNARY
