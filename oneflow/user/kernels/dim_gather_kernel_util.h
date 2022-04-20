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
#ifndef ONEFLOW_USER_KERNELS_DIM_GATHER_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_DIM_GATHER_KERNEL_UTIL_H_
#ifdef WITH_CUDA
#include "oneflow/core/cuda/atomic.cuh"
#endif  // WITH_CUDA
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

#define DIM_GATHER_SCATTER_DATA_TYPE_CPU_SEQ \
  ARITHMETIC_DATA_TYPE_SEQ                   \
  UNSIGNED_INT_DATA_TYPE_SEQ                 \
  BOOL_DATA_TYPE_SEQ

#define DIM_GATHER_SCATTER_DATA_TYPE_CUDA_SEQ \
  DIM_GATHER_SCATTER_DATA_TYPE_CPU_SEQ        \
  FLOAT16_DATA_TYPE_SEQ

constexpr int kDimGatherMaxDimCount = 8;

template<typename T>
using DimOpIndexNdHelper = NdIndexOffsetHelper<T, kDimGatherMaxDimCount>;

namespace user_op {

template<DeviceType device_type, typename IN_T, typename IDX_T>
struct DimGatherFunctor final {
  void operator()(ep::Stream* stream, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim, int64_t elem_cnt,
                  int32_t dim_length, int32_t dim, const IDX_T* index, const IN_T* input,
                  IN_T* output);
};

template<typename IN_T, typename IDX_T>
OF_DEVICE_FUNC void DoDimGather(const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                                const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim,
                                int64_t elem_cnt, int32_t dim_length, int32_t dim,
                                const IDX_T* index, const IN_T* input, IN_T* output) {
  XPU_1D_KERNEL_LOOP(index_offset, elem_cnt) {
    IDX_T coordinate[kDimGatherMaxDimCount] = {0};
    const IDX_T x = index[index_offset];
#ifdef __CUDA_ARCH__
    assert(x < dim_length && "gather index is out of bounds");
#else
    CHECK_LE(x, dim_length) << "RuntimeError: index " << x << " is out of bounds for dimension "
                            << dim << " with size " << dim_length;
#endif
    index_nd_helper.OffsetToNdIndex(index_offset, coordinate, ndim);
    coordinate[dim] = x;

    IDX_T input_offset = input_nd_helper.NdIndexToOffset(coordinate, ndim);
    output[index_offset] = input[input_offset];
  }
}

template<typename T>
struct DeviceAdd {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) {
#ifdef __CUDA_ARCH__
    cuda::atomic::Add(y, *x);  // TODO:(YaoChi), refine add using float16 -> half -> float -> half
#else
    *y += *x;
#endif
  };
};

// macros for functors instantiate(used by dim_gather_kernel_util.cu and dim_gather_kernel_uti.cpp)
#define INSTANTIATE_DIM_GATHER_FUNCTOR(device_type_v, dtype_pair, itype_pair)   \
  template struct DimGatherFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                   OF_PP_PAIR_FIRST(itype_pair)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DIM_GATHER_KERNEL_UTIL_H_
