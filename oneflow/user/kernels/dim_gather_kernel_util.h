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
#ifndef ONEFLOW_USER_KERNELS_ND_INDEX_SLICE_UTIL_H_
#define ONEFLOW_USER_KERNELS_ND_INDEX_SLICE_UTIL_H_
#include "oneflow/core/kernel/util/cuda_kernel_util.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

#define MAX_DIM_COUNT 8

template<typename T>
using DimOpIndexNdHelper = NdIndexOffsetHelper<T, MAX_DIM_COUNT>;

template<typename IDX_T>
struct NdIndexArg {
  static const unsigned int MAX_AXIS = MAX_DIM_COUNT;

  NdIndexArg(const ShapeView& tensorShape) : num_axis(tensorShape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, MAX_AXIS) {
      shape[i] = 0;
      coordinate[i] = 0;
    }

    FOR_RANGE(int64_t, i, 0, num_axis) { shape[i] = tensorShape.At(i); }
  }

  IDX_T shape[MAX_AXIS];
  IDX_T coordinate[MAX_AXIS];
  int64_t num_axis;
};

namespace user_op {
template<typename IN_T, typename IDX_T>
OF_DEVICE_FUNC void DoDimGather(NdIndexArg<IDX_T> inputArg, NdIndexArg<IDX_T> indexArg,
                                int64_t elem_cnt, int64_t dim, const IDX_T* index,
                                const IN_T* input, IN_T* output) {
  XPU_1D_KERNEL_LOOP(index_offset, elem_cnt) {
    DimOpIndexNdHelper<IDX_T> inputHelper(inputArg.shape, inputArg.num_axis);
    DimOpIndexNdHelper<IDX_T> indexHelper(indexArg.shape, indexArg.num_axis);

    // output[i][j][k] = input[i][x][k] # dim == 1, x = index[i][j][k]
    // output.shape == index.shape
    const IDX_T x = index[index_offset];
    indexHelper.OffsetToNdIndex(index_offset, inputArg.coordinate);
    inputArg.coordinate[dim] = x;

    IDX_T input_offset = inputHelper.NdIndexToOffset(inputArg.coordinate, inputArg.num_axis);
    output[index_offset] = input[input_offset];
  }
}

template<DeviceType device_type, typename T>
struct DeviceAdd {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) { *y += *x; };
};

template<DeviceType device_type, typename IN_T, typename IDX_T>
OF_DEVICE_FUNC void DoDimScatterAdd(NdIndexArg<IDX_T> srcArg, NdIndexArg<IDX_T> outputArg,
                                    int64_t elem_cnt, int64_t dim, const IDX_T* index,
                                    const IN_T* src, IN_T* output) {
  XPU_1D_KERNEL_LOOP(src_offset, elem_cnt) {
    // output[x][j][k] = src[i][j][k]  # if dim == 0, x = index[i][j][k]
    // index.shape == src.shape

    DimOpIndexNdHelper<IDX_T> srcHelper(srcArg.shape, srcArg.num_axis);
    DimOpIndexNdHelper<IDX_T> outputHelper(outputArg.shape, outputArg.num_axis);
    srcHelper.OffsetToNdIndex(src_offset, outputArg.coordinate);
    outputArg.coordinate[dim] = index[src_offset];  // x == index[src_offset]

    IDX_T output_offset = outputHelper.NdIndexToOffset(outputArg.coordinate, outputArg.num_axis);
    DeviceAdd<device_type, IN_T>::Invoke(src + src_offset, output + output_offset);
  }
}

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_ND_INDEX_SLICE_UTIL_H_
