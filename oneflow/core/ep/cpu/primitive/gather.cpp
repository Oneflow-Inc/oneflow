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
#include "oneflow/core/ep/include/primitive/gather.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/ep/common/primitive/util.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace {
template<typename T, typename K>
void GatherCpuKernel(T* src, T* dst, K* indice, const size_t num_indices, const size_t src_dim0,
                     const size_t src_dim1, const size_t src_dim2, const size_t offset) {
  FOR_RANGE(int64_t, outer_idx, 0, src_dim0) {
    FOR_RANGE(int64_t, i, 0, num_indices) {
      CHECK_GE(indice[i], 0);
      const int64_t idx = indice[i] - offset;
      T* to = dst + outer_idx * num_indices * src_dim2 + i * src_dim2;
      if (idx >= 0 && idx < src_dim1) {
        const T* from = src + outer_idx * src_dim1 * src_dim2 + idx * src_dim2;
        std::copy(from, from + src_dim2, to);
      } else {
        std::memset(reinterpret_cast<void*>(to), 0, src_dim2 * sizeof(T));
      }
    }
  }
}
template<typename T, typename K>
void BatchGatherCpuKernel(T* src, T* dst, K* indice, const size_t num_indices,
                          const size_t dst_dim0, const size_t dst_dim1, const size_t dst_dim2) {}

template<typename T, typename K, GatherKind gather_kind>
class GatherImpl : public Gather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() = default;
  void Launch(Stream* stream, const void* src, void* dst, const void* indice,
              const size_t num_indices, const size_t src_dim0, const size_t src_dim1,
              const size_t src_dim2, const size_t offset) override {
    if (gather_kind == GatherKind::kGather) {
      GatherCpuKernel(const_cast<T*>(static_cast<const T*>(src)), static_cast<T*>(dst),
                      static_cast<const K*>(indice), num_indices, src_dim0, src_dim1, src_dim2,
                      offset);
    } else if (gather_kind == GatherKind::kBatchGather) {
    }
  }
};
template<typename T, typename K, GatherKind gather_kind>
std::unique_ptr<Gather> NewGather() {
  return std::unique_ptr<Gather>(new GatherImpl<T, K, gather_kind>());
}
class GatherFactoryImpl : public GatherFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactoryImpl);
  GatherFactoryImpl() = default;
  ~GatherFactoryImpl() = default;
  std::unique_ptr<Gather> New(std::tuple<DataType, DataType> type_tuple,
                              GatherKind gather_kind) override {
#define MAKE_NEW_GATHER_ENTRY(in_type_pair, indice_type_pair)                             \
  {std::make_tuple(OF_PP_PAIR_SECOND(in_type_pair), OF_PP_PAIR_SECOND(indice_type_pair)), \
   NewGather<OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(indice_type_pair),          \
             GatherKind::kGather>},

    static const std::map<std::tuple<DataType, DataType>, std::function<std::unique_ptr<Gather>()>>
        new_gather_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_NEW_GATHER_ENTRY, GATHER_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)};

#undef MAKE_NEW_GATHER_ENTRY

#define MAKE_NEW_BATCH_GATHER_ENTRY(out_type_pair, indice_type_pair)                       \
  {std::make_tuple(OF_PP_PAIR_SECOND(out_type_pair), OF_PP_PAIR_SECOND(indice_type_pair)), \
   NewGather<OF_PP_PAIR_FIRST(out_type_pair), OF_PP_PAIR_FIRST(indice_type_pair),          \
             GatherKind::kBatchGather>},

    static const std::map<std::tuple<DataType, DataType>, std::function<std::unique_ptr<Gather>()>>
        new_batch_gather_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_NEW_BATCH_GATHER_ENTRY, GATHER_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)};

#undef MAKE_NEW_BATCH_GATHER_ENTRY

    if (gather_kind == GatherKind::kGather) {
      return NewPrimitiveFromHandlers(new_gather_handle, type_tuple);
    } else if (gather_kind == GatherKind::kBatchGather) {
      return NewPrimitiveFromHandlers(new_batch_gather_handle, type_tuple);
    } else {
      return nullptr;
    }
  }
};
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, GatherFactory, GatherFactoryImpl);
}  // namespace
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
