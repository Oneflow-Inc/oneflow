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

namespace {

using cuda::elementwise::GetNumBlocks;
using cuda::elementwise::kBlockSize;

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

  const auto* cond_pack = reinterpret_cast<const Packed<CondT, cond_pack_size>*>(params.cond);
  const auto* x_pack = reinterpret_cast<const Packed<T, x_pack_size>*>(params.x);
  const auto* y_pack = reinterpret_cast<const Packed<T, y_pack_size>*>(params.y);
  auto* z_pack = reinterpret_cast<Packed<T, pack_size>*>(params.z);

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
#pragma unroll
    for (size_t j = 0; j < pack_size; ++j) {
      const CondT cond_val =
          cond_pack_one ? cond_pack[cond_offset].elem[0] : cond_pack[cond_offset].elem[j];
      const T x_val = x_pack_one ? x_pack[x_offset].elem[0] : x_pack[x_offset].elem[j];
      const T y_val = y_pack_one ? y_pack[y_offset].elem[0] : y_pack[y_offset].elem[j];
      z_pack[offset].elem[j] = where_fn(cond_val, x_val, y_val);
    }
  }
}

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
cudaError_t LaunchCudaKernel(cudaStream_t stream, const int64_t* cond_dims, const int64_t* x_dims,
                             const int64_t* y_dims, const int64_t* z_dims, const CondT* cond,
                             const T* x, const T* y, T* z) {
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

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
void LaunchKernel(Stream* stream, const int64_t* cond_dims, const int64_t* x_dims,
                  const int64_t* y_dims, const int64_t* z_dims, const CondT* cond, const T* x,
                  const T* y, T* z) {
  static_assert(ndim > 0, "");
  auto cuda_stream = stream->As<CudaStream>()->cuda_stream();
  OF_CUDA_CHECK((LaunchCudaKernel<T, CondT, IndexT, ndim, cond_pack_size, x_pack_size, y_pack_size>(
      cuda_stream, cond_dims, x_dims, y_dims, z_dims, cond, x, y, z)));
}

template<typename T, typename CondT>
void LaunchScalarKernel(Stream* stream, const CondT* cond, const T* x, const T* y, T* z) {
  // should dispatch to elemwise tenary
  UNIMPLEMENTED();
}

void LaunchElemwiseTenary(CudaStream* stream, int64_t elem_cnt, DataType cond_type,
                          DataType data_type, const void* cond, const void* x, const void* y,
                          void* z) {
  const size_t data_type_size = GetSizeOfDataType(data_type);
  cudaStream_t cuda_stream = stream->cuda_stream();

#define IF(ctype, dtype_size)                                                      \
  if (cond_type == ctype && data_type_size == dtype_size) {                        \
    using T = typename std::aligned_storage<dtype_size, dtype_size>::type;         \
    using CondT = DataTypeToType<ctype>;                                           \
    WhereElemwiseFunctor<T, CondT, T, T> where_fn{};                               \
    OF_CUDA_CHECK((cuda::elementwise::Ternary<decltype(where_fn), T, CondT, T, T>( \
        where_fn, elem_cnt, static_cast<T*>(z), static_cast<const CondT*>(cond),   \
        static_cast<const T*>(x), static_cast<const T*>(y), cuda_stream)));        \
  }
#define ELIF(ctype, dtype_size) else IF(ctype, dtype_size)
#define ELSE         \
  else {             \
    UNIMPLEMENTED(); \
  }

  IF(DataType::kBool, 1)
  ELIF(DataType::kBool, 2)
  ELIF(DataType::kBool, 4)
  ELIF(DataType::kBool, 8)
  ELIF(DataType::kInt32, 1)
  ELIF(DataType::kInt32, 2)
  ELIF(DataType::kInt32, 4)
  ELIF(DataType::kInt32, 8)
  ELIF(DataType::kInt64, 1)
  ELIF(DataType::kInt64, 2)
  ELIF(DataType::kInt64, 4)
  ELIF(DataType::kInt64, 8)
  ELSE

#undef IF
#undef ELIF
#undef ELSE
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
      const size_t elem_cnt = GetElementCount(compact_ndim, compact_z_dims);
      LaunchElemwiseTenary(stream->As<CudaStream>(), elem_cnt, cond_type, data_type, cond, x, y, z);
    } else {
      // broadcast
      LaunchByDispatchType(stream, compact_ndim, compact_cond_dims, compact_x_dims, compact_y_dims,
                           compact_z_dims, cond_type, data_type, cond, x, y, z);
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

}  // namespace

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
