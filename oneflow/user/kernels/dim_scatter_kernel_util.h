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
#ifndef ONEFLOW_USER_KERNELS_DIM_SCATTER_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_DIM_SCATTER_KERNEL_UTIL_H_
#ifdef WITH_CUDA
#include "oneflow/core/cuda/atomic.cuh"
#endif  // WITH_CUDA

#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/error.pb.h"

namespace oneflow {

namespace user_op {

constexpr int kDimGatherMaxDimCount = 8;

template<typename T>
using DimOpIndexNdHelper = NdIndexOffsetHelper<T, kDimGatherMaxDimCount>;

#define INSTANTIATE_DIM_SCATTER_FUNCTORS(device_type, opt)               \
  template struct DimScatterFunctor<device_type, bool, int32_t, opt>;    \
  template struct DimScatterFunctor<device_type, uint8_t, int32_t, opt>; \
  template struct DimScatterFunctor<device_type, int8_t, int32_t, opt>;  \
  template struct DimScatterFunctor<device_type, int32_t, int32_t, opt>; \
  template struct DimScatterFunctor<device_type, int64_t, int32_t, opt>; \
  template struct DimScatterFunctor<device_type, float, int32_t, opt>;   \
  template struct DimScatterFunctor<device_type, double, int32_t, opt>;  \
  template struct DimScatterFunctor<device_type, bool, int64_t, opt>;    \
  template struct DimScatterFunctor<device_type, uint8_t, int64_t, opt>; \
  template struct DimScatterFunctor<device_type, int8_t, int64_t, opt>;  \
  template struct DimScatterFunctor<device_type, int32_t, int64_t, opt>; \
  template struct DimScatterFunctor<device_type, int64_t, int64_t, opt>; \
  template struct DimScatterFunctor<device_type, float, int64_t, opt>;   \
  template struct DimScatterFunctor<device_type, double, int64_t, opt>;

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

template<>
struct BinOpAddFunctor<bool> {
  OF_DEVICE_FUNC static void apply(const bool* x, bool* y) { *y += *x; }
};

template<>
struct BinOpAddFunctor<int8_t> {
  OF_DEVICE_FUNC static void apply(const int8_t* x, int8_t* y) { *y += *x; }
};

template<>
struct BinOpAddFunctor<uint8_t> {
  OF_DEVICE_FUNC static void apply(const uint8_t* x, uint8_t* y) { *y += *x; }
};

template<>
struct BinOpAddFunctor<int64_t> {
  OF_DEVICE_FUNC static void apply(const int64_t* x, int64_t* y) { *y += *x; }
};

template<typename T>
struct BinOpUpdateFunctor {
  OF_DEVICE_FUNC static void apply(const T* x, T* y) { *y = *x; }
};

template<DeviceType device_type, typename IN_T, typename IDX_T, template<typename T> class Opt>
struct DimScatterFunctor final {
  void operator()(ep::Stream* stream, const DimOpIndexNdHelper<IDX_T>& src_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,
                  const int64_t elem_cnt, const int32_t dim, const int64_t upper_bound,
                  const IDX_T* index, const IN_T* src, IN_T* output);
};

template<typename IN_T, typename IDX_T, template<typename T> class Opt>
OF_DEVICE_FUNC void DoDimScatter(const DimOpIndexNdHelper<IDX_T>& src_nd_helper,
                                 const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,
                                 const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,
                                 const int64_t elem_cnt, const int32_t dim, int64_t upper_bound,
                                 const IDX_T* index, const IN_T* src, IN_T* output) {
  XPU_1D_KERNEL_LOOP(idx_offset, elem_cnt) {
    IDX_T coordinate[kDimGatherMaxDimCount] = {0};
    idx_nd_helper.OffsetToNdIndex(idx_offset, coordinate, ndim);  // idx_offset -> ijk
    IDX_T idx_elem = index[idx_offset];
    if (upper_bound != 0 && idx_elem >= upper_bound) {
#if __CUDA_ARCH__
      __trap();
#else
      UNIMPLEMENTED() << "The index element " << idx_elem << " is out of bounds for dimension "
                      << dim << " with size " << upper_bound << ".";
#endif
    }
    IDX_T src_offset = src_nd_helper.NdIndexToOffset(coordinate, ndim);
    coordinate[dim] = idx_elem;
    IDX_T output_offset = output_nd_helper.NdIndexToOffset(coordinate, ndim);
    Opt<IN_T>::apply(src + src_offset, output + output_offset);
  }
}

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DIM_SCATTER_KERNEL_UTIL_H_
