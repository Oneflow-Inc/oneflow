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
#include "oneflow/user/kernels/dim_gather_scatter_util.h"

namespace oneflow {

namespace user_op {

DECLARE_DIMSCATTER_FUNCTOR(Add);
DECLARE_DIMSCATTER_FUNCTOR(Update);

template<typename IN_T, typename IDX_T>
OF_DEVICE_FUNC void DoDimScatterBinOp(const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                                      const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim,
                                      int64_t elem_cnt, int32_t dim, const IDX_T* index,
                                      const IN_T* input, IN_T* output, BinaryOpFn<IN_T> bin_op) {
  XPU_1D_KERNEL_LOOP(input_offset, elem_cnt) {
    IDX_T coordinate[kDimGatherMaxDimCount] = {0};
    input_nd_helper.OffsetToNdIndex(input_offset, coordinate, ndim);
    coordinate[dim] = index[input_offset];

    IDX_T output_offset = output_nd_helper.NdIndexToOffset(coordinate, ndim);
    bin_op(input + input_offset, output + output_offset);
  }
}

#define INSTANTIATE_DIM_SCATTER_ADD_FUNCTOR(device_type_v, dtype_pair, itype_pair)  \
  template struct DimScatterAddFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                       OF_PP_PAIR_FIRST(itype_pair)>;
#define INSTANTIATE_DIM_SCATTER_UPDATE_FUNCTOR(device_type_v, dtype_pair, itype_pair)  \
  template struct DimScatterUpdateFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                          OF_PP_PAIR_FIRST(itype_pair)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DIM_GATHER_KERNEL_UTIL_H_
