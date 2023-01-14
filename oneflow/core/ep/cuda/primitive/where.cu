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
#include "oneflow/core/ep/include/primitive/where.h"
#include "oneflow/core/ep/common/primitive/where.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {
namespace ep {
namespace primitive {

namespace where_cuda_impl {

using cuda::elementwise::GetNumBlocks;
using cuda::elementwise::kBlockSize;
using where_impl::BroadcastElementwiseWhereParams;
using where_impl::GetCompactBroadcastDims;
using where_impl::IsDimsEquals;
using where_impl::Pack;
using where_impl::PackType;
using where_impl::WhereFunctor;

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
__global__ void BroadcastElementwiseWhereCudaKernel(
    BroadcastElementwiseWhereParams<ndim, IndexT> params) {
  constexpr size_t _pack_size = (x_pack_size > y_pack_size) ? x_pack_size : y_pack_size;
  constexpr size_t pack_size = (cond_pack_size > _pack_size) ? cond_pack_size : _pack_size;
  static_assert(cond_pack_size == pack_size || cond_pack_size == 1, "");
  static_assert(x_pack_size == pack_size || x_pack_size == 1, "");
  static_assert(y_pack_size == pack_size || y_pack_size == 1, "");
  constexpr bool cond_pack_one = !(cond_pack_size == pack_size);
  constexpr bool x_pack_one = !(x_pack_size == pack_size);
  constexpr bool y_pack_one = !(y_pack_size == pack_size);

  const auto* cond = reinterpret_cast<const PackType<CondT, cond_pack_size>*>(params.cond);
  const auto* x = reinterpret_cast<const PackType<T, x_pack_size>*>(params.x);
  const auto* y = reinterpret_cast<const PackType<T, y_pack_size>*>(params.y);
  auto* z = reinterpret_cast<PackType<T, pack_size>*>(params.z);

  IndexT cond_index[ndim];
  IndexT x_index[ndim];
  IndexT y_index[ndim];
  IndexT z_index[ndim];

  WhereFunctor<T, CondT> where_fn{};

  CUDA_1D_KERNEL_LOOP_T(IndexT, offset, params.elem_cnt) {
    params.z_index_helper.OffsetToNdIndex(offset, z_index);
#pragma unroll
    for (size_t i = 0; i < ndim; ++i) {
      cond_index[i] = params.cond_index_mask[i] * z_index[i];
      x_index[i] = params.x_index_mask[i] * z_index[i];
      y_index[i] = params.y_index_mask[i] * z_index[i];
    }
    const IndexT cond_offset = params.cond_index_helper.NdIndexToOffset(cond_index);
    const IndexT x_offset = params.x_index_helper.NdIndexToOffset(x_index);
    const IndexT y_offset = params.y_index_helper.NdIndexToOffset(y_index);

    Pack<CondT, cond_pack_size> cond_pack;
    Pack<T, x_pack_size> x_pack;
    Pack<T, y_pack_size> y_pack;
    cond_pack.storage = cond[cond_offset];
    x_pack.storage = x[x_offset];
    y_pack.storage = y[y_offset];

    Pack<T, pack_size> z_pack;
#pragma unroll
    for (size_t j = 0; j < pack_size; ++j) {
      const CondT cond_val = cond_pack_one ? cond_pack.elem[0] : cond_pack.elem[j];
      const T x_val = x_pack_one ? x_pack.elem[0] : x_pack.elem[j];
      const T y_val = y_pack_one ? y_pack.elem[0] : y_pack.elem[j];
      z_pack.elem[j] = where_fn(cond_val, x_val, y_val);
    }
    z[offset] = z_pack.storage;
  }
}

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
cudaError_t LaunchCudaKernel(cudaStream_t stream, const int64_t* cond_dims, const int64_t* x_dims,
                             const int64_t* y_dims, const int64_t* z_dims, const void* cond,
                             const void* x, const void* y, void* z) {
  BroadcastElementwiseWhereParams<ndim, IndexT> params;
  params.cond_index_helper = NdIndexOffsetHelper<IndexT, ndim>(cond_dims);
  params.x_index_helper = NdIndexOffsetHelper<IndexT, ndim>(x_dims);
  params.y_index_helper = NdIndexOffsetHelper<IndexT, ndim>(y_dims);
  params.z_index_helper = NdIndexOffsetHelper<IndexT, ndim>(z_dims);
  for (size_t i = 0; i < ndim; ++i) {
    params.cond_index_mask[i] = (cond_dims[i] == 1) ? 0 : 1;
    params.x_index_mask[i] = (x_dims[i] == 1) ? 0 : 1;
    params.y_index_mask[i] = (y_dims[i] == 1) ? 0 : 1;
  }
  params.elem_cnt = static_cast<IndexT>(GetElementCount(ndim, z_dims));
  params.cond = cond;
  params.x = x;
  params.y = y;
  params.z = z;

  int num_blocks;
  {
    cudaError_t err = GetNumBlocks(params.elem_cnt, &num_blocks);
    if (err != cudaSuccess) { return err; }
  }
  BroadcastElementwiseWhereCudaKernel<T, CondT, IndexT, ndim, cond_pack_size, x_pack_size,
                                      y_pack_size><<<num_blocks, kBlockSize, 0, stream>>>(params);
  return cudaPeekAtLastError();
}

class WhereCudaImpl : public Where {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereCudaImpl);
  explicit WhereCudaImpl() = default;
  ~WhereCudaImpl() override = default;

  void Launch(Stream* stream, DataType cond_type, size_t cond_ndim, const int64_t* cond_dims,
              const void* cond, DataType data_type, size_t x_ndim, const int64_t* x_dims,
              const void* x, size_t y_ndim, const int64_t* y_dims, const void* y,
              void* z) override {
    size_t compact_ndim = 0;
    int64_t compact_cond_dims[kMaxNumDims] = {};
    int64_t compact_x_dims[kMaxNumDims] = {};
    int64_t compact_y_dims[kMaxNumDims] = {};
    int64_t compact_z_dims[kMaxNumDims] = {};
    GetCompactBroadcastDims(cond_ndim, cond_dims, x_ndim, x_dims, y_ndim, y_dims, compact_ndim,
                            compact_cond_dims, compact_x_dims, compact_y_dims, compact_z_dims);

    if (IsDimsEquals(compact_ndim, compact_z_dims, compact_cond_dims)
        && IsDimsEquals(compact_ndim, compact_z_dims, compact_x_dims)
        && IsDimsEquals(compact_ndim, compact_z_dims, compact_y_dims)) {
      // elementwise
      // using T = typename std::aligned_storage<data_type_size, data_type_size>::type;
      // using CondT = typename std::aligned_storage<cond_type_size, cond_type_size>::type;
      // const size_t elem_cnt = GetElementCount(compact_ndim, compact_z_dims);
      // auto cuda_stream = stream->As<CudaStream>()->cuda_stream();
      // OF_CUDA_CHECK(cuda::elementwise::Ternary(WhereFunctor<T>(), elem_cnt, z, cond, x, y,
      // cuda_stream));
    } else {
      // broadcast
      where_impl::LaunchByDispatchType(stream, compact_ndim, compact_cond_dims, compact_x_dims,
                                       compact_y_dims, compact_z_dims, cond_type, data_type, cond,
                                       x, y, z);
    }
  }
};

class WhereFactoryCudaImpl : public WhereFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereFactoryCudaImpl);
  WhereFactoryCudaImpl() = default;
  ~WhereFactoryCudaImpl() override = default;

  std::unique_ptr<Where> New() override { return std::unique_ptr<Where>(new WhereCudaImpl()); }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, WhereFactory, WhereFactoryCudaImpl);

}  // namespace where_cuda_impl

namespace where_impl {

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
void LaunchKernel(Stream* stream, const int64_t* cond_dims, const int64_t* x_dims,
                  const int64_t* y_dims, const int64_t* z_dims, const void* cond, const void* x,
                  const void* y, void* z) {
  auto cuda_stream = stream->As<CudaStream>()->cuda_stream();
  OF_CUDA_CHECK((where_cuda_impl::LaunchCudaKernel<T, CondT, IndexT, ndim, cond_pack_size,
                                                   x_pack_size, y_pack_size>(
      cuda_stream, cond_dims, x_dims, y_dims, z_dims, cond, x, y, z)));
}

}  // namespace where_impl

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
