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
#include <cstdint>
#ifdef WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/dim_gather_kernel_util.h"

namespace oneflow {

namespace user_op {

template<typename IN_T, typename IDX_T>
__global__ void DoCUDADimGather(const DimOpIndexNdHelper<IDX_T> input_nd_helper,
                                const DimOpIndexNdHelper<IDX_T> index_nd_helper, int ndim,
                                int64_t elem_cnt, int32_t dim, const IDX_T* index,
                                const IN_T* input, IN_T* output) {
  DoDimGather<IN_T, IDX_T>(input_nd_helper, index_nd_helper, ndim, elem_cnt, dim, index, input,
                           output);
}

template<typename IDX_T, typename IN_T>
struct DimGatherFunctor<DeviceType::kGPU, IN_T, IDX_T> final {
  void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim, int64_t elem_cnt,
                  int32_t dim, const IDX_T* index, const IN_T* input, IN_T* output) {
    RUN_CUDA_KERNEL((DoCUDADimGather<IN_T, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt),
                    input_nd_helper, index_nd_helper, ndim, elem_cnt, dim, index, input, output);
  }
};

// float16 special case of DimGatherFunctor template
template<typename IDX_T>
struct DimGatherFunctor<DeviceType::kGPU, float16, IDX_T> final {
  void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim, int64_t elem_cnt,
                  int32_t dim, const IDX_T* index, const float16* input, float16* output) {
    RUN_CUDA_KERNEL((DoCUDADimGather<half, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt),
                    input_nd_helper, index_nd_helper, ndim, elem_cnt, dim, index,
                    reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output));
  }
};

template<typename IN_T, typename IDX_T>
__global__ void DoCUDAScatterDimAdd(const DimOpIndexNdHelper<IDX_T> input_nd_helper,
                                    const DimOpIndexNdHelper<IDX_T> output_nd_helper, int ndim,
                                    int64_t elem_cnt, int32_t dim, const IDX_T* index,
                                    const IN_T* input, IN_T* output) {
  DoDimScatterAdd<IN_T, IDX_T>(input_nd_helper, output_nd_helper, ndim, elem_cnt, dim, index, input,
                               output);
}

template<typename IN_T, typename IDX_T>
struct DimScatterAddFunctor<DeviceType::kGPU, IN_T, IDX_T> final {
  void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim, int64_t elem_cnt,
                  int32_t dim, const IDX_T* index, const IN_T* input, IN_T* output) {
    RUN_CUDA_KERNEL((DoCUDAScatterDimAdd<IN_T, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt),
                    input_nd_helper, output_nd_helper, ndim, elem_cnt, dim, index, input, output);
  }
};

// float16 special case of DimScatterAddFunctor template
template<typename IDX_T>
struct DimScatterAddFunctor<DeviceType::kGPU, float16, IDX_T> final {
  void operator()(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim, int64_t elem_cnt,
                  int32_t dim, const IDX_T* index, const float16* input, float16* output) {
    RUN_CUDA_KERNEL((DoCUDAScatterDimAdd<half, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt),
                    input_nd_helper, output_nd_helper, ndim, elem_cnt, dim, index,
                    reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output));
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_DIM_GATHER_FUNCTOR, (DeviceType::kGPU),
                                 DIM_GATHER_SCATTER_DATA_TYPE_GPU_SEQ, INDEX_DATA_TYPE_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_DIM_SCATTER_ADD_FUNCTOR, (DeviceType::kGPU),
                                 DIM_GATHER_SCATTER_DATA_TYPE_GPU_SEQ, INDEX_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow

#endif  // WITH_CUDA
