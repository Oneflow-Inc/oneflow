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
#ifndef ONEFLOW_USER_KERNELS_DIM_GAHTER_SCATTER__UTIL_H_
#define ONEFLOW_USER_KERNELS_DIM_GAHTER_SCATTER__UTIL_H_

#ifdef WITH_CUDA
#include "oneflow/core/cuda/atomic.cuh"
#endif  // WITH_CUDA

#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

constexpr int kDimGatherMaxDimCount = 8;

namespace user_op {

template<typename T>
using DimOpIndexNdHelper = NdIndexOffsetHelper<T, kDimGatherMaxDimCount>;

template<typename T>
using BinaryOpFn = void (*)(const T* x, T* y);

template<typename T>
struct DeviceBinOp {
  OF_DEVICE_FUNC static void Add(const T* x, T* y) {
#ifdef __CUDA_ARCH__
    cuda::atomic::Add(y, *x);
#else
    *y += *x;
#endif
  }
  OF_DEVICE_FUNC static void Update(const T* x, T* y) { *y = *x; }
};

template<typename T>
struct BinOpAddFunctor {
  OF_DEVICE_FUNC static void apply(const T* x, T* y) {
#ifdef __CUDA_ARCH__
    cuda::atomic::Add(y, *x);
#else
    *y += *x;
#endif
  }
};

template<typename T>
struct BinOpUpdateFunctor {
  OF_DEVICE_FUNC static void Update(const T* x, T* y) { *y = *x; }
};

// ----- macros for scatter functors -----
#define DECLARE_DIMSCATTER_FUNCTOR(binop)                                                 \
  template<DeviceType device_type, typename IN_T, typename IDX_T>                         \
  struct DimScatter##binop##Functor final {                                               \
    void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& src_nd_helper,       \
                    const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,                       \
                    const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,    \
                    const int64_t elem_cnt, const int32_t dim, const int64_t upper_bound, \
                    const IDX_T* index, const IN_T* src, IN_T* output);                   \
  }

#define IMPLEMENT_DIMSCATTER_CPUFUNCTOR(binop)                                             \
  template<typename IN_T, typename IDX_T>                                                  \
  struct DimScatter##binop##Functor<DeviceType::kCPU, IN_T, IDX_T> final {                 \
    void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& src_nd_helper,        \
                    const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,                        \
                    const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,     \
                    const int64_t elem_cnt, const int32_t dim, const int64_t upper_bound,  \
                    const IDX_T* index, const IN_T* src, IN_T* output) {                   \
      DoDimScatterBinOp<IN_T, IDX_T>(src_nd_helper, idx_nd_helper, output_nd_helper, ndim, \
                                     elem_cnt, dim, upper_bound, index, src, output,       \
                                     DeviceBinOp<IN_T>::binop);                            \
    }                                                                                      \
  }

#define IMPLEMENT_DIMSCATTER_GPUFUNCTOR(binop)                                                     \
  template<typename IN_T, typename IDX_T>                                                          \
  __global__ void DoCUDADimScatter##binop(const DimOpIndexNdHelper<IDX_T> src_nd_helper,           \
                                          const DimOpIndexNdHelper<IDX_T> idx_nd_helper,           \
                                          const DimOpIndexNdHelper<IDX_T> output_nd_helper,        \
                                          const int ndim, const int64_t elem_cnt,                  \
                                          const int32_t dim, const int64_t upper_bound,            \
                                          const IDX_T* index, const IN_T* src, IN_T* output) {     \
    DoDimScatterBinOp<IN_T, IDX_T>(src_nd_helper, idx_nd_helper, output_nd_helper, ndim, elem_cnt, \
                                   dim, upper_bound, index, src, output,                           \
                                   DeviceBinOp<IN_T>::binop);                                      \
  }                                                                                                \
  template<typename IN_T, typename IDX_T>                                                          \
  struct DimScatter##binop##Functor<DeviceType::kGPU, IN_T, IDX_T> final {                         \
    void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& src_nd_helper,                \
                    const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,                                \
                    const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,             \
                    const int64_t elem_cnt, const int32_t dim, const int64_t upper_bound,          \
                    const IDX_T* index, const IN_T* src, IN_T* output) {                           \
      RUN_CUDA_KERNEL((DoCUDADimScatter##binop<IN_T, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt), \
                      src_nd_helper, idx_nd_helper, output_nd_helper, ndim, elem_cnt, dim,         \
                      upper_bound, index, src, output);                                            \
    }                                                                                              \
  };                                                                                               \
  template<typename IDX_T>                                                                         \
  struct DimScatter##binop##Functor<DeviceType::kGPU, float16, IDX_T> final {                      \
    void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& src_nd_helper,                \
                    const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,                                \
                    const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,             \
                    const int64_t elem_cnt, const int32_t dim, const int64_t upper_bound,          \
                    const IDX_T* index, const float16* src, float16* output) {                     \
      RUN_CUDA_KERNEL((DoCUDADimScatter##binop<half, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt), \
                      src_nd_helper, idx_nd_helper, output_nd_helper, ndim, elem_cnt, dim,         \
                      upper_bound, index, reinterpret_cast<const half*>(src),                      \
                      reinterpret_cast<half*>(output));                                            \
    }                                                                                              \
  }

}  // namespace user_op
}  // namespace oneflow

#endif
