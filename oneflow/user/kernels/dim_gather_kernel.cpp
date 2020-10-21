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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/user/kernels/dim_gather_kernel_util.h"
#include "oneflow/user/kernels/dim_gather_kernels.h"

namespace oneflow {

namespace user_op {

template<typename IN_T, typename IDX_T>
struct DimGatherFunctor<DeviceType::kCPU, IN_T, IDX_T> final {
  void operator()(NdIndexArg<IDX_T> inputArg, NdIndexArg<IDX_T> indexArg, int64_t elem_cnt,
                  int64_t dim, const IDX_T* index, const IN_T* input, IN_T* output,
                  DeviceCtx* ctx) {
    DoDimGather<IN_T, IDX_T>(inputArg, indexArg, elem_cnt, dim, index, input, output);
  }
};

template<typename IN_T, typename IDX_T>
struct DimScatterAddFunctor<DeviceType::kCPU, IN_T, IDX_T> final {
  void operator()(NdIndexArg<IDX_T> srcArg, NdIndexArg<IDX_T> outputArg, int64_t elem_cnt,
                  int64_t dim, const IDX_T* index, const IN_T* src, IN_T* output, DeviceCtx* ctx) {
    DoDimScatterAdd<IN_T, IDX_T>(srcArg, outputArg, elem_cnt, dim, index, src,
                                                   output);
  }
};

#define REGISTER_DIM_GATHER_KERNEL(device, dtype_pair, itype_pair)                               \
  REGISTER_USER_KERNEL("dim_gather")                                                            \
      .SetCreateFn<                                                                             \
          DimGatherKernel<device, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(itype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("input", 0) == OF_PP_PAIR_SECOND(dtype_pair))       \
                       & (user_op::HobDataType("index", 0) == OF_PP_PAIR_SECOND(itype_pair)));

#define REGISTER_DIM_SCATTER_KERNEL(device, dtype_pair, itype_pair)                               \
  REGISTER_USER_KERNEL("dim_scatter_add_like")                                                   \
      .SetCreateFn<                                                                              \
          ScatterDimKernel<device, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(itype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                       \
                       & (user_op::HobDataType("input", 0) == OF_PP_PAIR_SECOND(dtype_pair))          \
                       & (user_op::HobDataType("index", 0) == OF_PP_PAIR_SECOND(itype_pair)));


OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DIM_GATHER_KERNEL, (DeviceType::kCPU),
                                 DIM_GATHER_SCATTER_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DIM_SCATTER_KERNEL, (DeviceType::kCPU),
                                 DIM_GATHER_SCATTER_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
