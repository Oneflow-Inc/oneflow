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
#include "oneflow/core/kernel/util/cuda_kernel_util.h"
#endif  // WITH_CUDA
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

#define DIM_GATHER_SCATTER_DATA_TYPE_CPU_SEQ \
  FLOATING_DATA_TYPE_SEQ                     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define DIM_GATHER_SCATTER_DATA_TYPE_GPU_SEQ \
  DIM_GATHER_SCATTER_DATA_TYPE_CPU_SEQ       \
  FLOAT16_DATA_TYPE_SEQ

constexpr int kDimGatherMaxDimCount = 8;

template<typename T>
using DimOpIndexNdHelper = NdIndexOffsetHelper<T, kDimGatherMaxDimCount>;

namespace user_op {

template<DeviceType device_type, typename IN_T, typename IDX_T>
struct DimGatherFunctor final {
  void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim, int64_t elem_cnt,
                  int32_t dim, const IDX_T* index, const IN_T* input, IN_T* output);
};

template<DeviceType device_type, typename IN_T, typename IDX_T>
struct DimScatterAddFunctor final {
  void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim, int64_t elem_cnt,
                  int32_t dim, const IDX_T* index, const IN_T* src, IN_T* output);
};

template<typename IN_T, typename IDX_T>
OF_DEVICE_FUNC void DoDimGather(const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                                const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim,
                                int64_t elem_cnt, int32_t dim, const IDX_T* index,
                                const IN_T* input, IN_T* output) {
  XPU_1D_KERNEL_LOOP(index_offset, elem_cnt) {
    IDX_T coordinate[kDimGatherMaxDimCount] = {0};
    const IDX_T x = index[index_offset];
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
    gpu_atomic_add(y, *x);  // TODO:(YaoChi), refine add using float16 -> half -> float -> half
#else
    *y += *x;
#endif
  };
};

template<typename IN_T, typename IDX_T>
OF_DEVICE_FUNC void DoDimScatterAdd(const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                                    const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim,
                                    int64_t elem_cnt, int32_t dim, const IDX_T* index,
                                    const IN_T* input, IN_T* output) {
  XPU_1D_KERNEL_LOOP(input_offset, elem_cnt) {
    IDX_T coordinate[kDimGatherMaxDimCount] = {0};
    input_nd_helper.OffsetToNdIndex(input_offset, coordinate, ndim);
    coordinate[dim] = index[input_offset];

    IDX_T output_offset = output_nd_helper.NdIndexToOffset(coordinate, ndim);
    DeviceAdd<IN_T>::Invoke(input + input_offset, output + output_offset);
  }
}

// macros for functors instantiate(used by dim_gather_kernel_util.cu and dim_gather_kernel_uti.cpp)
#define INSTANTIATE_DIM_GATHER_FUNCTOR(device_type_v, dtype_pair, itype_pair)   \
  template struct DimGatherFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                   OF_PP_PAIR_FIRST(itype_pair)>;

#define INSTANTIATE_DIM_SCATTER_ADD_FUNCTOR(device_type_v, dtype_pair, itype_pair)  \
  template struct DimScatterAddFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                       OF_PP_PAIR_FIRST(itype_pair)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DIM_GATHER_KERNEL_UTIL_H_
