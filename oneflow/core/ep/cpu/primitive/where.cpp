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
#include "oneflow/core/ep/cpu/cpu_stream.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace {

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
void BroadcastElementwiseWhereKernel(CpuStream* cpu_stream,
                                     const BroadcastElementwiseWhereParams<ndim, IndexT>& params) {
  constexpr size_t _pack_size = (x_pack_size > y_pack_size) ? x_pack_size : y_pack_size;
  constexpr size_t pack_size = (cond_pack_size > _pack_size) ? cond_pack_size : _pack_size;
  static_assert(cond_pack_size == pack_size || cond_pack_size == 1, "");
  static_assert(x_pack_size == pack_size || x_pack_size == 1, "");
  static_assert(y_pack_size == pack_size || y_pack_size == 1, "");

  const auto* cond_pack = reinterpret_cast<const Packed<CondT, cond_pack_size>*>(params.cond);
  const auto* x_pack = reinterpret_cast<const Packed<T, x_pack_size>*>(params.x);
  const auto* y_pack = reinterpret_cast<const Packed<T, y_pack_size>*>(params.y);
  auto* z_pack = reinterpret_cast<Packed<T, pack_size>*>(params.z);

  WhereFunctor<T, CondT> where_fn{};

  cpu_stream->ParallelFor(0, params.elem_cnt, [&](int64_t begin, int64_t end) {
    IndexT cond_index[ndim];
    IndexT x_index[ndim];
    IndexT y_index[ndim];
    IndexT z_index[ndim];

    for (IndexT offset = begin; offset < end; offset++) {
      params.z_index_helper.OffsetToNdIndex(offset, z_index);
      for (size_t i = 0; i < ndim; ++i) {
        cond_index[i] = params.cond_index_mask[i] * z_index[i];
        x_index[i] = params.x_index_mask[i] * z_index[i];
        y_index[i] = params.y_index_mask[i] * z_index[i];
      }
      const IndexT cond_offset = params.cond_index_helper.NdIndexToOffset(cond_index);
      const IndexT x_offset = params.x_index_helper.NdIndexToOffset(x_index);
      const IndexT y_offset = params.y_index_helper.NdIndexToOffset(y_index);

      for (size_t j = 0; j < pack_size; ++j) {
        const CondT cond_val = (cond_pack_size == pack_size) ? cond_pack[cond_offset].elem[j]
                                                             : cond_pack[cond_offset].elem[0];
        const T x_val =
            (x_pack_size == pack_size) ? x_pack[x_offset].elem[j] : x_pack[x_offset].elem[0];
        const T y_val =
            (y_pack_size == pack_size) ? y_pack[y_offset].elem[j] : y_pack[y_offset].elem[0];
        z_pack[offset].elem[j] = where_fn(static_cast<bool>(cond_val), x_val, y_val);
      }
    }
  });
}

template<typename T, typename CondT>
void ScalarWhereKernel(const CondT* cond, const T* x, const T* y, T* z) {
  WhereFunctor<T, CondT> where_fn{};
  *z = where_fn(*cond, *x, *y);
}

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
void LaunchKernel(Stream* stream, const int64_t* cond_dims, const int64_t* x_dims,
                  const int64_t* y_dims, const int64_t* z_dims, const CondT* cond, const T* x,
                  const T* y, T* z) {
  static_assert(ndim > 0, "");
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

  auto* cpu_stream = stream->As<CpuStream>();
  BroadcastElementwiseWhereKernel<T, CondT, IndexT, ndim, cond_pack_size, x_pack_size, y_pack_size>(
      cpu_stream, params);
}

template<typename T, typename CondT>
void LaunchScalarKernel(Stream* stream, const CondT* cond, const T* x, const T* y, T* z) {
  ScalarWhereKernel(cond, x, y, z);
}

template<typename T, typename CondT>
class WhereImpl : public Where {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereImpl);
  explicit WhereImpl() = default;
  ~WhereImpl() override = default;

  void Launch(Stream* stream, size_t num_cond_dims, const int64_t* cond_dims, const void* cond,
              size_t num_x_dims, const int64_t* x_dims, const void* x, size_t num_y_dims,
              const int64_t* y_dims, const void* y, void* z) override {
    size_t compact_num_dims = 0;
    int64_t compact_cond_dims[kMaxNumDims] = {};
    int64_t compact_x_dims[kMaxNumDims] = {};
    int64_t compact_y_dims[kMaxNumDims] = {};
    int64_t compact_z_dims[kMaxNumDims] = {};
    GetCompactBroadcastDims(num_cond_dims, cond_dims, num_x_dims, x_dims, num_y_dims, y_dims,
                            &compact_num_dims, compact_cond_dims, compact_x_dims, compact_y_dims,
                            compact_z_dims);
    LaunchByDispatchNDim(stream, compact_num_dims, compact_cond_dims, compact_x_dims,
                         compact_y_dims, compact_z_dims, static_cast<const CondT*>(cond),
                         static_cast<const T*>(x), static_cast<const T*>(y), static_cast<T*>(z));
  }
};

class WhereFactoryImpl : public WhereFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereFactoryImpl);
  WhereFactoryImpl() = default;
  ~WhereFactoryImpl() override = default;

  std::unique_ptr<Where> New(DataType cond_type, DataType data_type, size_t max_num_dims) override {
    return NewWhere<WhereImpl>(cond_type, data_type, max_num_dims);
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, WhereFactory, WhereFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
