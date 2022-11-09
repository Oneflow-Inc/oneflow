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
__global__ void CudaGatherNd(NdIndexSliceArgs args, const I* indices, const T* dense, T* slices) {
  DoGatherNd(args.num_slices * args.slice_size, args.slice_size, args.index_ndims, args.dense_shape,
             indices, dense, slices);
}

template<typename T, typename I>
__global__ void CudaScatterNdAdd(NdIndexSliceArgs args, const I* indices, const T* slices,
                                 T* dense) {
  DoScatterNdAdd<DeviceType::kCUDA>(args.num_slices * args.slice_size, args.slice_size,
                                    args.index_ndims, args.dense_shape, indices, slices, dense);
}

template<typename T, typename I>
__global__ void CudaScatterNdUpdate(NdIndexSliceArgs args, const I* indices, const T* slices,
                                    T* dense) {
  DoScatterNdUpdate<DeviceType::kCUDA>(args.num_slices * args.slice_size, args.slice_size,
                                       args.index_ndims, args.dense_shape, indices, slices, dense);
}

template<typename T, typename I>
__global__ void CudaScatterNdUpdateWithStride(NdIndexSliceArgs args, const I* indices,
                                              const T* slices, T* dense) {
  DoScatterNdUpdateWithStride<DeviceType::kCUDA>(args.num_slices * args.slice_size, args, indices,
                                                 slices, dense);
}

template<typename T, typename I>
__global__ void CudaFillByNdIndex(NdIndexSliceArgs args, const I* indices, T* dense, T value) {
  DoFillByNdIndex(args.num_slices * args.slice_size, args.slice_size, args.index_ndims,
                  args.dense_shape, indices, dense, value);
}

}  // namespace

template<typename T, typename I>
struct GatherNdFunctor<DeviceType::kCUDA, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* dense, T* slices) const {
    RUN_CUDA_KERNEL((CudaGatherNd<T, I>), stream, args.num_slices * args.slice_size, args, indices,
                    dense, slices);
  }
};

template<typename T, typename I>
struct ScatterNdAddFunctor<DeviceType::kCUDA, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* slices, T* dense) const {
    RUN_CUDA_KERNEL((CudaScatterNdAdd<T, I>), stream, args.num_slices * args.slice_size, args,
                    indices, slices, dense);
  }
};

template<typename T, typename I>
struct ScatterNdUpdateFunctor<DeviceType::kCUDA, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* slices, T* dense) const {
    RUN_CUDA_KERNEL((CudaScatterNdUpdate<T, I>), stream, args.num_slices * args.slice_size, args,
                    indices, slices, dense);
  }
};

template<typename T, typename I>
struct ScatterNdUpdateWithStrideFunctor<DeviceType::kCUDA, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* slices, T* dense) const {
    RUN_CUDA_KERNEL((CudaScatterNdUpdateWithStride<T, I>), stream,
                    args.num_slices * args.slice_size, args, indices, slices, dense);
  }
};

template<typename T, typename I>
struct FillByNdIndexFunctor<DeviceType::kCUDA, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices, T* dense,
                  T value) const {
    RUN_CUDA_KERNEL((CudaFillByNdIndex<T, I>), stream, args.num_slices * args.slice_size, args,
                    indices, dense, value);
  }
};

template<typename T>
struct DeviceAdd<DeviceType::kCUDA, T> {
  __device__ __forceinline__ static void Invoke(const T* x, T* y) { cuda::atomic::Add(y, *x); }
};

template<>
struct DeviceAdd<DeviceType::kCUDA, bool> {
  __device__ __forceinline__ static void Invoke(const bool* x, bool* y) { *y += *x; }
};

template<>
struct DeviceAdd<DeviceType::kCUDA, uint8_t> {
  __device__ __forceinline__ static void Invoke(const uint8_t* x, uint8_t* y) { *y += *x; }
};

template<>
struct DeviceAdd<DeviceType::kCUDA, int8_t> {
  __device__ __forceinline__ static void Invoke(const int8_t* x, int8_t* y) { *y += *x; }
};

template<>
struct DeviceAdd<DeviceType::kCUDA, int64_t> {
  __device__ __forceinline__ static void Invoke(const int64_t* x, int64_t* y) { *y += *x; }
};

#define CUDA_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ                        \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    INSTANTIATE_GATHER_ND_FUNCTOR, (DeviceType::kCUDA),
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SCATTER_ND_ADD_FUNCTOR, (DeviceType::kCUDA),
                                 CUDA_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_FILL_BY_ND_INDEX_FUNCTOR, (DeviceType::kCUDA),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_GATHER_ND_KERNELS, (DeviceType::kCUDA),
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_SCATTER_ND_KERNELS, (DeviceType::kCUDA),
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCATTER_ND_LIKE_KERNELS, (DeviceType::kCUDA),
                                 CUDA_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_TENSOR_GATHER_ND_UPDATE_KERNELS, (DeviceType::kCUDA),
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_TENSOR_GATHER_ND_ADD_KERNELS, (DeviceType::kCUDA),
                                 CUDA_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

template<>
struct DeviceAdd<DeviceType::kCUDA, float16> {
  __device__ __forceinline__ static void Invoke(const float16* x, float16* y) {
    cuda::atomic::Add(reinterpret_cast<half*>(y), *(reinterpret_cast<const half*>(x)));
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ND_INDEX_SLICE_FUNCTORS, (DeviceType::kCUDA),
                                 FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ND_INDEX_SLICE_KERNELS, (DeviceType::kCUDA),
                                 FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct DeviceAdd<DeviceType::kCUDA, bfloat16> {
  __device__ __forceinline__ static void Invoke(const bfloat16* x, bfloat16* y) {
    cuda::atomic::Add(reinterpret_cast<nv_bfloat16*>(y),
                      *(reinterpret_cast<const nv_bfloat16*>(x)));
  }
};
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ND_INDEX_SLICE_FUNCTORS, (DeviceType::kCUDA),
                                 BFLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ND_INDEX_SLICE_KERNELS, (DeviceType::kCUDA),
                                 BFLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif

}  // namespace oneflow
