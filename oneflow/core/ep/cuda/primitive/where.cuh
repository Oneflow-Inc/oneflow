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
#ifndef ONEFLOW_CORE_EP_CUDA_PRIMITIVE_WHERE_H_
#define ONEFLOW_CORE_EP_CUDA_PRIMITIVE_WHERE_H_

#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/common/primitive/util.h"
#include "oneflow/core/ep/common/primitive/where.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace where_cuda_details {

using cuda::elementwise::GetNumBlocks;
using cuda::elementwise::kBlockSize;
using where_details::Pack;
using where_details::PackType;
using where_details::WhereFunctor;

template<size_t NDIM, typename IndexType>
struct BroadcastElementwiseWhereParams {
  NdIndexOffsetHelper<IndexType, NDIM> cond_index_helper;
  NdIndexOffsetHelper<IndexType, NDIM> x_index_helper;
  NdIndexOffsetHelper<IndexType, NDIM> y_index_helper;
  NdIndexOffsetHelper<IndexType, NDIM> z_index_helper;
  IndexType cond_index_mask[NDIM];
  IndexType x_index_mask[NDIM];
  IndexType y_index_mask[NDIM];
  IndexType elem_cnt{};
  const void* cond{};
  const void* x{};
  const void* y{};
  void* z{};
};

template<typename T, typename CondT, typename IndexType, size_t NDIM, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
__global__ void BroadcastElementwiseWhereCudaKernel(
    BroadcastElementwiseWhereParams<NDIM, IndexType> params) {
  constexpr size_t xy_max_pack_size = (x_pack_size > y_pack_size) ? x_pack_size : y_pack_size;
  constexpr size_t z_pack_size =
      (cond_pack_size > xy_max_pack_size) ? cond_pack_size : xy_max_pack_size;
  static_assert(cond_pack_size == z_pack_size || cond_pack_size == 1, "");
  static_assert(x_pack_size == z_pack_size || x_pack_size == 1, "");
  static_assert(y_pack_size == z_pack_size || y_pack_size == 1, "");

  IndexType cond_index[NDIM];
  IndexType x_index[NDIM];
  IndexType y_index[NDIM];
  IndexType z_index[NDIM];

  const auto* cond = reinterpret_cast<const PackType<CondT, cond_pack_size>*>(params.cond);
  const auto* x = reinterpret_cast<const PackType<T, x_pack_size>*>(params.x);
  const auto* y = reinterpret_cast<const PackType<T, y_pack_size>*>(params.y);
  auto* z = reinterpret_cast<PackType<T, z_pack_size>*>(params.z);

  CUDA_1D_KERNEL_LOOP_T(IndexType, offset, params.elem_cnt) {
    params.z_index_helper.OffsetToNdIndex(offset, z_index);
#pragma unroll
    for (int i = 0; i < NDIM; ++i) {
      cond_index[i] = params.cond_index_mask[i] * z_index[i];
      x_index[i] = params.x_index_mask[i] * z_index[i];
      y_index[i] = params.y_index_mask[i] * z_index[i];
    }
    const IndexType cond_offset = params.cond_index_helper.NdIndexToOffset(cond_index);
    const IndexType x_offset = params.x_index_helper.NdIndexToOffset(x_index);
    const IndexType y_offset = params.y_index_helper.NdIndexToOffset(y_index);

    Pack<CondT, cond_pack_size> cond_pack;
    Pack<T, x_pack_size> x_pack;
    Pack<T, y_pack_size> y_pack;
    cond_pack.storage = cond[cond_offset];
    x_pack.storage = x[x_offset];
    y_pack.storage = y[y_offset];

    Pack<T, z_pack_size> z_pack;
    WhereFunctor<T, CondT> where{};
#pragma unroll
    for (int j = 0; j < z_pack_size; ++j) {
      const CondT cond_val =
          (cond_pack_size == z_pack_size) ? cond_pack.elem[j] : cond_pack.elem[0];
      const T x_val = (x_pack_size == z_pack_size) ? x_pack.elem[j] : x_pack.elem[0];
      const T y_val = (y_pack_size == z_pack_size) ? y_pack.elem[j] : y_pack.elem[0];
      z_pack.elem[j] = where(cond_val, x_val, y_val);
    }
    z[offset] = z_pack.storage;
  }
}

template<typename T, typename CondT, typename IndexType, size_t NDIM, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
cudaError_t LaunchKernel(cudaStream_t stream, const int64_t* cond_dims, const CondT* cond,
                         const int64_t* x_dims, const T* x, const int64_t* y_dims, const T* y,
                         const int64_t* z_dims, T* z) {
  BroadcastElementwiseWhereParams<NDIM, IndexType> params;
  params.cond_index_helper = NdIndexOffsetHelper<IndexType, NDIM>(cond_dims);
  params.x_index_helper = NdIndexOffsetHelper<IndexType, NDIM>(x_dims);
  params.y_index_helper = NdIndexOffsetHelper<IndexType, NDIM>(y_dims);
  params.z_index_helper = NdIndexOffsetHelper<IndexType, NDIM>(z_dims);
  for (size_t i = 0; i < NDIM; ++i) {
    params.cond_index_mask[i] = (cond_dims[i] == 1) ? 0 : 1;
    params.x_index_mask[i] = (x_dims[i] == 1) ? 0 : 1;
    params.y_index_mask[i] = (y_dims[i] == 1) ? 0 : 1;
  }
  params.elem_cnt = static_cast<IndexType>(GetElementCount(NDIM, z_dims));
  params.cond = cond;
  params.x = x;
  params.y = y;
  params.z = z;

  int num_blocks;
  {
    cudaError_t err = GetNumBlocks(params.elem_cnt, &num_blocks);
    if (err != cudaSuccess) { return err; }
  }
  BroadcastElementwiseWhereCudaKernel<T, CondT, IndexType, NDIM, cond_pack_size, x_pack_size,
                                      y_pack_size><<<num_blocks, kBlockSize, 0, stream>>>(params);
  return cudaPeekAtLastError();
}

template<typename T, typename CondT, size_t NDIM, size_t cond_pack_size, size_t x_pack_size,
         size_t y_pack_size>
cudaError_t LaunchByDispatchIndexType(cudaStream_t stream, const int64_t* cond_dims,
                                      const CondT* cond, const int64_t* x_dims, const T* x,
                                      const int64_t* y_dims, const T* y, const int64_t* z_dims,
                                      T* z) {
  const size_t elem_cnt = GetElementCount(NDIM, z_dims);
  if (elem_cnt < GetMaxVal<int32_t>()) {
    return LaunchKernel<T, CondT, int32_t, NDIM, cond_pack_size, x_pack_size, y_pack_size>(
        stream, cond_dims, cond, x_dims, x, y_dims, y, z_dims, z);
  } else {
    return LaunchKernel<T, CondT, int64_t, NDIM, cond_pack_size, x_pack_size, y_pack_size>(
        stream, cond_dims, cond, x_dims, x, y_dims, y, z_dims, z);
  }
}

template<typename T, typename CondT, size_t cond_pack_size, size_t x_pack_size, size_t y_pack_size>
cudaError_t LaunchByDispatchNDim(cudaStream_t stream, size_t ndim, const int64_t* cond_dims,
                                 const CondT* cond, const int64_t* x_dims, const T* x,
                                 const int64_t* y_dims, const T* y, const int64_t* z_dims, T* z) {
  cudaError_t (*func)(cudaStream_t /*stream*/, const int64_t* /*cond_dims*/, const CondT* /*cond*/,
                      const int64_t* /*x_dims*/, const T* /*x*/, const int64_t* /*y_dims*/,
                      const T* /*y*/, const int64_t* /*z_dims*/, T* /*z*/) = nullptr;
  CHECK_GT(ndim, 0);
  if (ndim == 1) {
    func = LaunchByDispatchIndexType<T, CondT, 1, cond_pack_size, x_pack_size, y_pack_size>;
  } else if (ndim == 2) {
    func = LaunchByDispatchIndexType<T, CondT, 2, cond_pack_size, x_pack_size, y_pack_size>;
  } else if (ndim == 3) {
    func = LaunchByDispatchIndexType<T, CondT, 3, cond_pack_size, x_pack_size, y_pack_size>;
  } else if (ndim == 4) {
    func = LaunchByDispatchIndexType<T, CondT, 4, cond_pack_size, x_pack_size, y_pack_size>;
  } else {
    UNIMPLEMENTED();
  }
  return func(stream, cond_dims, cond, x_dims, x, y_dims, y, z_dims, z);
}

template<size_t max_pack_size, typename T, typename CondT>
size_t GetPackSize(size_t ndim, const int64_t* cond_dims, const CondT* cond, const int64_t* x_dims,
                   const T* x, const int64_t* y_dims, const T* y, const int64_t* z_dims,
                   const T* z) {
  static_assert(max_pack_size > 0 && (max_pack_size & (max_pack_size - 1)) == 0, "");
  CHECK_NE(z_dims[ndim - 1], 1);
  for (size_t pack_size = max_pack_size; pack_size > 2; pack_size /= 2) {
    if (!IsPackSizeSupported<T>(pack_size, ndim, z_dims, z)) { continue; }
    if (x_dims[ndim - 1] != 1 && !IsPackSizeSupported<T>(pack_size, ndim, x_dims, x)) { continue; }
    if (y_dims[ndim - 1] != 1 && !IsPackSizeSupported<T>(pack_size, ndim, y_dims, y)) { continue; }
    if (cond_dims[ndim - 1] != 1 && !IsPackSizeSupported<CondT>(pack_size, ndim, cond_dims, cond)) {
      continue;
    }
    return pack_size;
  }
  return 1;
}

template<typename T, typename CondT>
cudaError_t LaunchByDispatchPackSize(cudaStream_t stream, size_t ndim, int64_t* cond_dims,
                                     const CondT* cond, int64_t* x_dims, const T* x,
                                     int64_t* y_dims, const T* y, int64_t* z_dims, T* z) {
  constexpr size_t kMaxPackSize = 4;
  size_t pack_size =
      GetPackSize<kMaxPackSize, T, CondT>(ndim, cond_dims, cond, x_dims, x, y_dims, y, z_dims, z);
  size_t cond_pack_size = 1;
  size_t x_pack_size = 1;
  size_t y_pack_size = 1;
  if (pack_size > 1) {
    if (cond_dims[ndim - 1] != 1) {
      cond_dims[ndim - 1] /= pack_size;
      cond_pack_size = pack_size;
    }
    if (x_dims[ndim - 1] != 1) {
      x_dims[ndim - 1] /= pack_size;
      x_pack_size = pack_size;
    }
    if (y_dims[ndim - 1] != 1) {
      y_dims[ndim - 1] /= pack_size;
      y_pack_size = pack_size;
    }
    z_dims[ndim - 1] /= pack_size;
  }

  cudaError_t (*func)(cudaStream_t /*stream*/, size_t /*ndim*/, const int64_t* /*cond_dims*/,
                      const CondT* /*cond*/, const int64_t* /*x_dims*/, const T* /*x*/,
                      const int64_t* /*y_dims*/, const T* /*y*/, const int64_t* /*z_dims*/,
                      T* /*z*/) = nullptr;

#define ELIF(c, x, y)                                                     \
  else if (cond_pack_size == c && x_pack_size == x && y_pack_size == y) { \
    func = LaunchByDispatchNDim<T, CondT, c, x, y>;                       \
  }

  if (pack_size == 1) { func = LaunchByDispatchNDim<T, CondT, 1, 1, 1>; }
  ELIF(4, 4, 4)
  ELIF(4, 4, 1)
  ELIF(4, 1, 4)
  ELIF(4, 1, 1)
  ELIF(4, 4, 1)
  ELIF(1, 4, 4)
  ELIF(1, 4, 1)
  ELIF(4, 1, 4)
  ELIF(1, 4, 4)
  ELIF(1, 1, 4)
  else {
    UNIMPLEMENTED();
  }
#undef ELIF
  return func(stream, ndim, cond_dims, cond, x_dims, x, y_dims, y, z_dims, z);
}

template<typename T, typename CondT>
cudaError_t Launch(cudaStream_t stream, size_t cond_ndim, const int64_t* cond_dims,
                   const CondT* cond, size_t x_ndim, const int64_t* x_dims, const T* x,
                   size_t y_ndim, const int64_t* y_dims, const T* y, T* z) {
  size_t compact_ndim = 0;
  int64_t compact_cond_dims[kMaxNumDims] = {};
  int64_t compact_x_dims[kMaxNumDims] = {};
  int64_t compact_y_dims[kMaxNumDims] = {};
  int64_t compact_z_dims[kMaxNumDims] = {};
  where_details::GetCompactBroadcastDims(cond_ndim, cond_dims, x_ndim, x_dims, y_ndim, y_dims,
                                         compact_ndim, compact_cond_dims, compact_x_dims,
                                         compact_y_dims, compact_z_dims);

  if (where_details::IsDimsEquals(compact_ndim, compact_z_dims, compact_cond_dims)
      && where_details::IsDimsEquals(compact_ndim, compact_z_dims, compact_x_dims)
      && where_details::IsDimsEquals(compact_ndim, compact_z_dims, compact_y_dims)) {
    // elementwise
    const size_t elem_cnt = GetElementCount(compact_ndim, compact_z_dims);
    return cuda::elementwise::Ternary(WhereFunctor<T, CondT>(), elem_cnt, z, cond, x, y, stream);
  } else {
    // broadcast
    return LaunchByDispatchPackSize(stream, compact_ndim, compact_cond_dims, cond, compact_x_dims,
                                    x, compact_y_dims, y, compact_z_dims, z);
  }
}

}  // namespace where_cuda_details

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_CUDA_PRIMITIVE_WHERE_H_
