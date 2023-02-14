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
#ifndef ONEFLOW_USER_KERNELS_DIM_SCATTER_SCALAR_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_DIM_SCATTER_SCALAR_KERNEL_UTIL_H_
#ifdef WITH_CUDA
#include "oneflow/core/cuda/atomic.cuh"
#include <cuda_fp16.h>
#endif  // WITH_CUDA
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

#define NO_HALF_UTIL_FOUND         \
  printf("cuda arch must >= 530"); \
  assert(false)

namespace user_op {

constexpr int kDimGatherMaxDimCount = 8;

template<typename T>
struct AddScalarFunctor {
  OF_DEVICE_FUNC static void apply(const T x, T* y) {
#ifdef __CUDA_ARCH__
    cuda::atomic::Add(y, x);
#else
    *y += x;
#endif
  }
};

#ifdef WITH_CUDA
template<>
struct AddScalarFunctor<half> {
  OF_DEVICE_FUNC static void apply(const half x, half* y) {
#if __CUDA_ARCH__
    *y = __float2half(__half2float(*y) + __half2float(x));
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
#endif

template<>
struct AddScalarFunctor<int8_t> {
  OF_DEVICE_FUNC static void apply(const int8_t x, int8_t* y) { *y += x; }
};

template<>
struct AddScalarFunctor<uint8_t> {
  OF_DEVICE_FUNC static void apply(const uint8_t x, uint8_t* y) { *y += x; }
};

template<>
struct AddScalarFunctor<int64_t> {
  OF_DEVICE_FUNC static void apply(const int64_t x, int64_t* y) { *y += x; }
};

template<typename T>
struct UpdateScalarFunctor {
  OF_DEVICE_FUNC static void apply(const T x, T* y) { *y = x; }
};

#define INSTANTIATE_DIM_SCATTER_SCARLAR_CPU_FUNCTORS(device_type, opt)         \
  template struct DimScatterScalarFunctor<device_type, uint8_t, int32_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, int8_t, int32_t, opt>;  \
  template struct DimScatterScalarFunctor<device_type, int32_t, int32_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, int64_t, int32_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, float, int32_t, opt>;   \
  template struct DimScatterScalarFunctor<device_type, double, int32_t, opt>;  \
  template struct DimScatterScalarFunctor<device_type, float16, int32_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, uint8_t, int64_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, int8_t, int64_t, opt>;  \
  template struct DimScatterScalarFunctor<device_type, int32_t, int64_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, int64_t, int64_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, float, int64_t, opt>;   \
  template struct DimScatterScalarFunctor<device_type, double, int64_t, opt>;  \
  template struct DimScatterScalarFunctor<device_type, float16, int64_t, opt>;

#define INSTANTIATE_DIM_SCATTER_SCARLAR_CUDA_FUNCTORS(device_type, opt)        \
  template struct DimScatterScalarFunctor<device_type, uint8_t, int32_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, int8_t, int32_t, opt>;  \
  template struct DimScatterScalarFunctor<device_type, int32_t, int32_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, int64_t, int32_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, float, int32_t, opt>;   \
  template struct DimScatterScalarFunctor<device_type, double, int32_t, opt>;  \
  template struct DimScatterScalarFunctor<device_type, half, int32_t, opt>;    \
  template struct DimScatterScalarFunctor<device_type, uint8_t, int64_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, int8_t, int64_t, opt>;  \
  template struct DimScatterScalarFunctor<device_type, int32_t, int64_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, int64_t, int64_t, opt>; \
  template struct DimScatterScalarFunctor<device_type, float, int64_t, opt>;   \
  template struct DimScatterScalarFunctor<device_type, double, int64_t, opt>;  \
  template struct DimScatterScalarFunctor<device_type, half, int64_t, opt>;

template<typename T>
using DimOpIndexNdHelper = NdIndexOffsetHelper<T, kDimGatherMaxDimCount>;

template<DeviceType device_type, typename IN_T, typename IDX_T, template<typename T> class Opt>
struct DimScatterScalarFunctor final {
  void operator()(ep::Stream* stream, const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,
                  const int64_t elem_cnt, const int32_t dim, int64_t upper_bound,
                  const IDX_T* index, const IN_T src, IN_T* output);
};

template<typename IN_T, typename IDX_T, template<typename T> class Opt>
OF_DEVICE_FUNC void DoScatterScalarFunctor(const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,
                                           const DimOpIndexNdHelper<IDX_T>& output_nd_helper,
                                           const int ndim, const int64_t elem_cnt,
                                           const int32_t dim, int64_t upper_bound,
                                           const IDX_T* index, const IN_T src, IN_T* output) {
  XPU_1D_KERNEL_LOOP(idx_offset, elem_cnt) {
    IDX_T coordinate[kDimGatherMaxDimCount] = {0};

    idx_nd_helper.OffsetToNdIndex(idx_offset, coordinate, ndim);  // idx_offset -> ijk
    IDX_T idx_elem = index[idx_offset];
    if (idx_elem >= upper_bound) {
#if __CUDA_ARCH__
      __trap();
#else
      UNIMPLEMENTED() << "The index element " << idx_elem << " is out of bounds for dimension "
                      << dim << " with size " << upper_bound << ".";
#endif
    }
    coordinate[dim] = idx_elem;
    IDX_T output_offset = output_nd_helper.NdIndexToOffset(coordinate, ndim);
    Opt<IN_T>::apply(src, output + output_offset);
  }
}

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DIM_SCATTER_SCALAR_KERNEL_UTIL_H_
