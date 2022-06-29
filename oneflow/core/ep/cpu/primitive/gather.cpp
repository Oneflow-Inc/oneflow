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
class GatherImpl : public Gather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() = default;
  void Launch(Stream* stream, const void* src, void* dst, const void* indice,
              const size_t num_indices, const size_t src_dim0, const size_t src_dim1,
              const size_t src_dim2, const size_t offset) override {
    K* indice_ = reinterpret_cast<K*>(indice);
    T* src_ = reinterpret_cast<T*>(src);
    T* dst_ = reinterpret_cast<T*>(dst);
    FOR_RANGE(int64_t, outer_idx, 0, src_dim0) {
      FOR_RANGE(int64_t, i, 0, num_indices) {
        CHECK_GE(indice_[i], 0);
        const int64_t idx = indice_[i] - offset;
        T* to = dst_ + outer_idx * num_indices * src_dim2 + i * src_dim2;
        if (idx >= 0 && idx < src_dim1) {
          const T* from = src_ + outer_idx * src_dim1 * src_dim2 + idx * src_dim2;
          std::copy(from, from + src_dim2, to);
        } else {
          std::memset(reinterpret_cast<void*>(to), 0, src_dim2 * sizeof(T));
        }
      }
    }
  }
};
template<typename T, typename K>
std::unique_ptr<Gather> NewGather() {
  return std::unique_ptr<Gather>(new GatherImpl<T, K>());
}
class GatherFactoryImpl : public GatherFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactoryImpl);
  GatherFactoryImpl() = default;
  ~GatherFactoryImpl() = default;
  std::unique_ptr<Gather> New(DataType data_type) override {
#define MAKE_NEW_GATHER_ENTRY(in_type, indice_type) \
  {OF_PP_PAIR_SECOND(in_type), NewGather<OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indice_type)>},

    static const std::map<DataType, std::function<std::unique_ptr<Gather>()>> new_gather_handle{
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_GATHER_ENTRY, CPU_PRIMITIVE_FLOATING_TYPE_SEQ)};

#undef MAKE_NEW_GATHER_ENTRY
    return NewPrimitiveFromHandlers(new_gather_handle, data_type);
  }
};
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, GatherFactory, GatherFactoryImpl);
}  // namespace
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
