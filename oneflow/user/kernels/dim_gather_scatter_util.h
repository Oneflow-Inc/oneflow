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
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

#define DIM_GATHER_SCATTER_DATA_TYPE_CPU_SEQ \
  FLOATING_DATA_TYPE_SEQ                     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define DIM_GATHER_SCATTER_DATA_TYPE_GPU_SEQ \
  DIM_GATHER_SCATTER_DATA_TYPE_CPU_SEQ       \
  FLOAT16_DATA_TYPE_SEQ

constexpr int kDimGatherMaxDimCount = 8;

namespace user_op {

template<typename T>
using DimOpIndexNdHelper = NdIndexOffsetHelper<T, kDimGatherMaxDimCount>;

template<typename T>
using BinaryOpFn = void (*)(const T* x, T* y);

// Steps for adding a binary operation on scatter are as follows:
// 1. implment binop in DeviceBinOp, for example "Mul":
//    OF_DEVICE_FUNC static void Mul(const T* x, T* y) { *y *= *x; }
// 2. Implement and register kernels in dim_scatter_kernels.cpp:
//    IMPLEMENT_AND_REGISTER_KERNEL("scatter_mul_like", Mul);
// 3. Declare Functor in dim_scatter_kernel_util.h:
//    DECLARE_DIMSCATTER_FUNCTOR(Mul);
// 4. Implement functors in dim_scatter_kernel_util.cu and cpp file:
//    in .cu file:
//      IMPLEMENT_DIMSCATTER_GPUFUNCTOR(Mul);
//    in .cpp file:
//      IMPLEMENT_DIMSCATTER_CPUFUNCTOR(Mul);
//

template<typename T>
struct DeviceBinOp {
  OF_DEVICE_FUNC static void Add(const T* x, T* y) {
#ifdef __CUDA_ARCH__
    gpu_atomic_add(y, *x);  // TODO:(YaoChi), refine add using float16 -> half -> float -> half
#else
    *y += *x;
#endif
  }

  OF_DEVICE_FUNC static void Update(const T* x, T* y) { *y = *x; }
};

#define DECLARE_DIMSCATTER_FUNCTOR(binop)                                                          \
  template<DeviceType device_type, typename IN_T, typename IDX_T>                                  \
  struct DimScatter##binop##Functor final {                                                        \
    void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,              \
                    const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim, int64_t elem_cnt, \
                    int32_t dim, const IDX_T* index, const IN_T* src, IN_T* output);               \
  }

#define IMPLEMENT_DIMSCATTER_CPUFUNCTOR(binop)                                                     \
  template<typename IN_T, typename IDX_T>                                                          \
  struct DimScatter##binop##Functor<DeviceType::kCPU, IN_T, IDX_T> final {                         \
    void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,              \
                    const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim, int64_t elem_cnt, \
                    int32_t dim, const IDX_T* index, const IN_T* input, IN_T* output) {            \
      DoDimScatterBinOp<IN_T, IDX_T>(input_nd_helper, output_nd_helper, ndim, elem_cnt, dim,       \
                                     index, input, output, DeviceBinOp<IN_T>::binop);              \
    }                                                                                              \
  }

#define IMPLEMENT_DIMSCATTER_GPUFUNCTOR(binop)                                                     \
  template<typename IN_T, typename IDX_T>                                                          \
  __global__ void DoCUDADimScatter##binop(const DimOpIndexNdHelper<IDX_T> input_nd_helper,         \
                                          const DimOpIndexNdHelper<IDX_T> output_nd_helper,        \
                                          int ndim, int64_t elem_cnt, int32_t dim,                 \
                                          const IDX_T* index, const IN_T* input, IN_T* output) {   \
    DoDimScatterBinOp<IN_T, IDX_T>(input_nd_helper, output_nd_helper, ndim, elem_cnt, dim, index,  \
                                   input, output, DeviceBinOp<IN_T>::binop);                       \
  }                                                                                                \
  template<typename IN_T, typename IDX_T>                                                          \
  struct DimScatter##binop##Functor<DeviceType::kGPU, IN_T, IDX_T> final {                         \
    void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,              \
                    const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim, int64_t elem_cnt, \
                    int32_t dim, const IDX_T* index, const IN_T* input, IN_T* output) {            \
      RUN_CUDA_KERNEL((DoCUDADimScatter##binop<IN_T, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt), \
                      input_nd_helper, output_nd_helper, ndim, elem_cnt, dim, index, input,        \
                      output);                                                                     \
    }                                                                                              \
  };                                                                                               \
  template<typename IDX_T>                                                                         \
  struct DimScatter##binop##Functor<DeviceType::kGPU, float16, IDX_T> final {                      \
    void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,              \
                    const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim, int64_t elem_cnt, \
                    int32_t dim, const IDX_T* index, const float16* input, float16* output) {      \
      RUN_CUDA_KERNEL((DoCUDADimScatter##binop<half, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt), \
                      input_nd_helper, output_nd_helper, ndim, elem_cnt, dim, index,               \
                      reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output));      \
    }                                                                                              \
  }

}  // namespace user_op
}  // namespace oneflow

#endif