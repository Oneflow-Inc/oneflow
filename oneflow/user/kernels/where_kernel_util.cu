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
#include "oneflow/user/kernels/where_kernel_util.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T, typename CondT>
struct WhereFunctor {
  OF_DEVICE_FUNC T operator()(CondT cond, T lhs, T rhs) const {
    return static_cast<bool>(cond) ? lhs : rhs;
  }
};

template<typename T, typename CondT>
struct WhereScalarXFunctor {
  OF_DEVICE_FUNC explicit WhereScalarXFunctor(T scalar) : x_scalar(scalar) {}
  OF_DEVICE_FUNC T operator()(CondT cond, T rhs) const {
    return static_cast<bool>(cond) ? x_scalar : rhs;
  }
  const T x_scalar;
};

template<typename T, typename CondT>
struct WhereScalarYFunctor {
  OF_DEVICE_FUNC explicit WhereScalarYFunctor(T scalar) : y_scalar(scalar) {}
  OF_DEVICE_FUNC T operator()(CondT cond, T lhs) const {
    return static_cast<bool>(cond) ? lhs : y_scalar;
  }
  const T y_scalar;
};

template<typename T, typename CondT>
struct WhereScalarXYFunctor {
  OF_DEVICE_FUNC explicit WhereScalarXYFunctor(T x_scalar, T y_scalar)
      : x_scalar(x_scalar), y_scalar(y_scalar) {}
  OF_DEVICE_FUNC T operator()(CondT cond) const {
    return static_cast<bool>(cond) ? x_scalar : y_scalar;
  }
  const T x_scalar;
  const T y_scalar;
};

}  // namespace

template<typename T, typename CondT>
struct WhereKernelUtil<DeviceType::kCUDA, T, CondT> {
  static void Where(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                    const T* rhs, T* out) {
    cuda::elementwise::Ternary(WhereFunctor<T, CondT>(), elem_cnt, out, cond, lhs, rhs,
                               stream->As<ep::CudaStream>()->cuda_stream());
  }
  static void WhereXScalar(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond,
                           const T x_scalar, const T* rhs, T* out) {
    cuda::elementwise::Binary(WhereScalarXFunctor<T, CondT>(x_scalar), elem_cnt, out, cond, rhs,
                              stream->As<ep::CudaStream>()->cuda_stream());
  }
  static void WhereYScalar(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond,
                           const T* lhs, const T y_scalar, T* out) {
    cuda::elementwise::Binary(WhereScalarYFunctor<T, CondT>(y_scalar), elem_cnt, out, cond, lhs,
                              stream->As<ep::CudaStream>()->cuda_stream());
  }
  static void WhereXYScalar(ep::Stream* stream, const int64_t elem_cnt, const CondT* cond,
                            const T x_scalar, const T y_scalar, T* out) {
    cuda::elementwise::Unary(WhereScalarXYFunctor<T, CondT>(x_scalar, y_scalar), elem_cnt, out,
                             cond, stream->As<ep::CudaStream>()->cuda_stream());
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTOR, (DeviceType::kCUDA),
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)

}  // namespace oneflow
