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
#include "oneflow/user/kernels/gather_dim_kernel_util.h"
#include "oneflow/user/kernels/gather_dim_kernels.h"

namespace oneflow {

namespace user_op {

template<>
struct DeviceAdd<DeviceType::kGPU, float16> {
  __device__ __forceinline__ static void Invoke(const float16* x, float16* y) {
    gpu_atomic_add(reinterpret_cast<half*>(y), *(reinterpret_cast<const half*>(x)));
  }
};

template<typename T>
struct DeviceAdd<DeviceType::kGPU, T> {
  __device__ __forceinline__ static void Invoke(const T* x, T* y) { gpu_atomic_add(y, *x); }
};

template<typename IN_T, typename IDX_T>
__global__ void DoCUDAGatherDim(NdIndexArg<IDX_T> inputArg, NdIndexArg<IDX_T> indexArg,
                                int64_t elem_cnt, int64_t dim, const IDX_T* index,
                                const IN_T* input, IN_T* output) {
  DoGatherDim<IN_T, IDX_T>(inputArg, indexArg, elem_cnt, dim, index, input, output);
}

template<typename IDX_T, typename IN_T>
struct GatherDimFunctor<DeviceType::kGPU, IN_T, IDX_T> final {
  void operator()(NdIndexArg<IDX_T> inputArg, NdIndexArg<IDX_T> indexArg, int64_t elem_cnt,
                  int64_t dim, const IDX_T* index, const IN_T* input, IN_T* output,
                  DeviceCtx* ctx) {
    RUN_CUDA_KERNEL((DoCUDAGatherDim<IN_T, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt), inputArg,
                    indexArg, elem_cnt, dim, index, input, output);
  }
};

template<typename IN_T, typename IDX_T>
__global__ void DoCUDAScatterDimAdd(NdIndexArg<IDX_T> srcArg, NdIndexArg<IDX_T> outputArg,
                                    int64_t elem_cnt, int64_t dim, const IDX_T* index,
                                    const IN_T* src, IN_T* output) {
  DoScatterDimAdd<DeviceType::kGPU, IN_T, IDX_T>(srcArg, outputArg, elem_cnt, dim, index, src,
                                                 output);
}

template<typename IN_T, typename IDX_T>
struct ScatterDimAddFunctor<DeviceType::kGPU, IN_T, IDX_T> final {
  void operator()(NdIndexArg<IDX_T> srcArg, NdIndexArg<IDX_T> outputArg, int64_t elem_cnt,
                  int64_t dim, const IDX_T* index, const IN_T* src, IN_T* output, DeviceCtx* ctx) {
    RUN_CUDA_KERNEL((DoCUDAScatterDimAdd<IN_T, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt), srcArg,
                    outputArg, elem_cnt, dim, index, src, output);
  }
};

// float16 special case of ScatterDimAddFunctor template
template<typename IDX_T>
struct ScatterDimAddFunctor<DeviceType::kGPU, float16, IDX_T> final {
  void operator()(NdIndexArg<IDX_T> srcArg, NdIndexArg<IDX_T> outputArg, int64_t elem_cnt,
                  int64_t dim, const IDX_T* index, const float16* src, float16* output,
                  DeviceCtx* ctx) {
    RUN_CUDA_KERNEL((DoCUDAScatterDimAdd<half, IDX_T>), ctx, BlocksNum4ThreadsNum(elem_cnt), srcArg,
                    outputArg, elem_cnt, dim, index, reinterpret_cast<const half*>(src),
                    reinterpret_cast<half*>(output));
  }
};

#define REGISTER_GATHERDIM_GPUKERNEL(device, in_type, indices_type)                             \
  REGISTER_USER_KERNEL("gather_dim")                                                            \
      .SetCreateFn<                                                                             \
          GatherDimKernel<device, OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("input", 0) == OF_PP_PAIR_SECOND(in_type))       \
                       & (user_op::HobDataType("index", 0) == OF_PP_PAIR_SECOND(indices_type)));

#define REGISTER_SCATTERDIMADD_GPUKERNEL(device, in_type, indices_type)                          \
  REGISTER_USER_KERNEL("scatter_dim_add_like")                                                   \
      .SetCreateFn<                                                                              \
          ScatterDimKernel<device, OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                       \
                       & (user_op::HobDataType("src", 0) == OF_PP_PAIR_SECOND(in_type))          \
                       & (user_op::HobDataType("index", 0) == OF_PP_PAIR_SECOND(indices_type)));

#define GATHER_SCATTER_DIM_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GATHERDIM_GPUKERNEL, (DeviceType::kGPU),
                                 GATHER_SCATTER_DIM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCATTERDIMADD_GPUKERNEL, (DeviceType::kGPU),
                                 GATHER_SCATTER_DIM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
