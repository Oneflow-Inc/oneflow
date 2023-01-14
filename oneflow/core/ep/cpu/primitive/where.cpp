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

namespace where_impl {

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
void BroadcastElementwiseWhereKernel(CpuStream* cpu_stream,
                                     const BroadcastElementwiseWhereParams<ndim, IndexT>& params) {
  constexpr size_t _pack_size = (x_pack_size > y_pack_size) ? x_pack_size : y_pack_size;
  constexpr size_t pack_size = (cond_pack_size > _pack_size) ? cond_pack_size : _pack_size;
  static_assert(cond_pack_size == pack_size || cond_pack_size == 1, "");
  static_assert(x_pack_size == pack_size || x_pack_size == 1, "");
  static_assert(y_pack_size == pack_size || y_pack_size == 1, "");

  const auto* cond = reinterpret_cast<const PackType<CondT, cond_pack_size>*>(params.cond);
  const auto* x = reinterpret_cast<const PackType<T, x_pack_size>*>(params.x);
  const auto* y = reinterpret_cast<const PackType<T, y_pack_size>*>(params.y);
  auto* z = reinterpret_cast<PackType<T, pack_size>*>(params.z);

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

      Pack<CondT, cond_pack_size> cond_pack;
      Pack<T, x_pack_size> x_pack;
      Pack<T, y_pack_size> y_pack;
      cond_pack.storage = cond[cond_offset];
      x_pack.storage = x[x_offset];
      y_pack.storage = y[y_offset];

      Pack<T, pack_size> z_pack;
      for (size_t j = 0; j < pack_size; ++j) {
        const CondT cond_val =
            (cond_pack_size == pack_size) ? cond_pack.elem[j] : cond_pack.elem[0];
        const T x_val = (x_pack_size == pack_size) ? x_pack.elem[j] : x_pack.elem[0];
        const T y_val = (y_pack_size == pack_size) ? y_pack.elem[j] : y_pack.elem[0];
        z_pack.elem[j] = where_fn(static_cast<bool>(cond_val), x_val, y_val);
      }
      z[offset] = z_pack.storage;
    }
  });
}

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
void LaunchKernel(Stream* stream, const int64_t* cond_dims, const int64_t* x_dims,
                  const int64_t* y_dims, const int64_t* z_dims, const void* cond, const void* x,
                  const void* y, void* z) {
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

class WhereImpl : public Where {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereImpl);
  explicit WhereImpl() = default;
  ~WhereImpl() override = default;

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
    LaunchByDispatchType(stream, compact_ndim, compact_cond_dims, compact_x_dims, compact_y_dims,
                         compact_z_dims, cond_type, data_type, cond, x, y, z);
  }
};

class WhereFactoryImpl : public WhereFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereFactoryImpl);
  WhereFactoryImpl() = default;
  ~WhereFactoryImpl() override = default;

  std::unique_ptr<Where> New() override { return std::unique_ptr<Where>(new WhereImpl()); }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, WhereFactory, WhereFactoryImpl);

}  // namespace where_impl

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
