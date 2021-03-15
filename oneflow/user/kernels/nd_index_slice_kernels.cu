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
#include "oneflow/user/kernels/nd_index_slice_kernels.h"
#include "oneflow/core/cuda/atomic.cuh"

namespace oneflow {

namespace {

template<typename T, typename I>
__global__ void CudaGatherNd(NdIndexSliceArgs<T, I> args, const I* indices, const T* dense,
                             T* slices) {
  DoGatherNd(args.num_slices * args.slice_size, args.slice_size, args.index_ndims, args.dense_shape,
             indices, dense, slices);
}

template<typename T, typename I>
__global__ void CudaScatterNdAdd(NdIndexSliceArgs<T, I> args, const I* indices, const T* slices,
                                 T* dense) {
  DoScatterNdAdd<DeviceType::kGPU>(args.num_slices * args.slice_size, args.slice_size,
                                   args.index_ndims, args.dense_shape, indices, slices, dense);
}

template<typename T, typename I>
__global__ void CudaZeroByNdIndex(NdIndexSliceArgs<T, I> args, const I* indices, T* dense) {
  DoZeroByNdIndex(args.num_slices * args.slice_size, args.slice_size, args.index_ndims,
                  args.dense_shape, indices, dense);
}

}  // namespace

template<typename T, typename I>
struct GatherNdFunctor<DeviceType::kGPU, T, I> final {
  void operator()(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  const T* dense, T* slices) const {
    RUN_CUDA_KERNEL((CudaGatherNd<T, I>), ctx, args.num_slices * args.slice_size, args, indices,
                    dense, slices);
  }
};

template<typename T, typename I>
struct ScatterNdAddFunctor<DeviceType::kGPU, T, I> final {
  void operator()(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  const T* slices, T* dense) const {
    RUN_CUDA_KERNEL((CudaScatterNdAdd<T, I>), ctx, args.num_slices * args.slice_size, args, indices,
                    slices, dense);
  }
};

template<typename T, typename I>
struct ZeroByNdIndexFunctor<DeviceType::kGPU, T, I> final {
  void operator()(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  T* dense) const {
    RUN_CUDA_KERNEL((CudaZeroByNdIndex<T, I>), ctx, args.num_slices * args.slice_size, args,
                    indices, dense);
  }
};

template<typename T>
struct DeviceAdd<DeviceType::kGPU, T> {
  __device__ __forceinline__ static void Invoke(const T* x, T* y) { cuda::atomic::Add(y, *x); }
};

#define GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ                       \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GATHER_ND_FUNCTOR, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SCATTER_ND_ADD_FUNCTOR, (DeviceType::kGPU),
                                 GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ZERO_BY_ND_INDEX_FUNCTOR, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GATHER_ND_KERNELS, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCATTER_ND_KERNELS, (DeviceType::kGPU),
                                 GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCATTER_ND_LIKE_KERNELS, (DeviceType::kGPU),
                                 GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_TENSOR_GATHER_ND_UPDATE_KERNELS, (DeviceType::kGPU),
                                 GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_TENSOR_GATHER_ND_ADD_KERNELS, (DeviceType::kGPU),
                                 GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000

template<>
struct DeviceAdd<DeviceType::kGPU, float16> {
  __device__ __forceinline__ static void Invoke(const float16* x, float16* y) {
    cuda::atomic::Add(reinterpret_cast<half*>(y), *(reinterpret_cast<const half*>(x)));
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ND_INDEX_SLICE_FUNCTORS, (DeviceType::kGPU),
                                 FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ND_INDEX_SLICE_KERNELS, (DeviceType::kGPU),
                                 FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#endif

}  // namespace oneflow
