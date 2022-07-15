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
void GatherCpuKernel(int64_t outer_dim_size, int64_t gather_dim_size, int64_t inner_dim_size,
                     const T* data, int64_t num_indices, const K* indice, int64_t offset,
                     T* output) {
  FOR_RANGE(int64_t, outer_idx, 0, outer_dim_size) {
    FOR_RANGE(int64_t, i, 0, num_indices) {
      CHECK_GE(indice[i], static_cast<K>(0));
      const int64_t idx = indice[i] - offset;
      T* to = output + outer_idx * num_indices * inner_dim_size + i * inner_dim_size;
      if (idx >= 0 && idx < gather_dim_size) {
        const T* from = data + outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size;
        std::copy(from, from + inner_dim_size, to);
      } else {
        std::memset(reinterpret_cast<void*>(to), 0, inner_dim_size * sizeof(T));
      }
    }
  }
}

template<typename T, typename K>
void BatchGatherCpuKernel(int64_t batch_dim_size, int64_t outer_dim_size, int64_t gather_dim_size,
                          int64_t inner_dim_size, const T* data, int64_t num_indices,
                          const K* indice, int64_t offset, T* output) {
  const int64_t indice_instance_size = num_indices / batch_dim_size;
  const int64_t data_instance_size = outer_dim_size * gather_dim_size * inner_dim_size;
  const int64_t output_instance_size = outer_dim_size * indice_instance_size * inner_dim_size;

  FOR_RANGE(int64_t, batch_idx, 0, batch_dim_size) {
    const T* batch_data = data + batch_idx * data_instance_size;
    T* batch_output = output + batch_idx * output_instance_size;
    const K* batch_indice = indice + batch_idx * indice_instance_size;

    GatherCpuKernel(outer_dim_size, gather_dim_size, inner_dim_size, batch_data,
                    indice_instance_size, batch_indice, offset, batch_output);
  }
}

template<typename T, typename K>
class GatherImpl : public Gather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() override = default;
  void Launch(Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
              int64_t gather_dim_size, int64_t inner_dim_size, const void* data,
              int64_t num_indices, const void* indice, void* output) override {
    if (batch_dim_size == 1) {
      GatherCpuKernel(outer_dim_size, gather_dim_size, inner_dim_size,
                      const_cast<T*>(static_cast<const T*>(data)), num_indices,
                      static_cast<const K*>(indice), /*offset*/ 0, static_cast<T*>(output));
    } else {
      BatchGatherCpuKernel(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                           const_cast<T*>(static_cast<const T*>(data)), num_indices,
                           static_cast<const K*>(indice), /*offset*/ 0, static_cast<T*>(output));
    }
  }
  void Launch(Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
              int64_t gather_dim_size, int64_t inner_dim_size, const void* data,
              int64_t num_indices, const void* indice, int64_t offset, void* output) override {
    if (batch_dim_size == 1) {
      GatherCpuKernel(outer_dim_size, gather_dim_size, inner_dim_size,
                      const_cast<T*>(static_cast<const T*>(data)), num_indices,
                      static_cast<const K*>(indice), offset, static_cast<T*>(output));
    } else {
      BatchGatherCpuKernel(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                           const_cast<T*>(static_cast<const T*>(data)), num_indices,
                           static_cast<const K*>(indice), offset, static_cast<T*>(output));
    }
  }
};

template<typename T, typename K>
std::unique_ptr<Gather> NewGather() {
  return std::unique_ptr<Gather>(new GatherImpl<T, K>());
}
#define GATHER_DATA_TYPE_SEQ ARITHMETIC_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(bool, DataType::kBool)
#define GATHER_INDEX_TYPE_SEQ INDEX_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32)
class GatherFactoryImpl : public GatherFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactoryImpl);
  GatherFactoryImpl() = default;
  ~GatherFactoryImpl() override = default;
  std::unique_ptr<Gather> New(DataType data_dtype, DataType indice_type) override {
    std::tuple<DataType, DataType> type_tuple = std::make_tuple(data_dtype, indice_type);
#define MAKE_NEW_GATHER_ENTRY(in_type_pair, indice_type_pair)                             \
  {std::make_tuple(OF_PP_PAIR_SECOND(in_type_pair), OF_PP_PAIR_SECOND(indice_type_pair)), \
   NewGather<OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(indice_type_pair)>},

    static const std::map<std::tuple<DataType, DataType>, std::function<std::unique_ptr<Gather>()>>
        new_gather_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_NEW_GATHER_ENTRY, GATHER_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)};

#undef MAKE_NEW_GATHER_ENTRY
    return NewPrimitiveFromHandlers(new_gather_handle, type_tuple);
  }
};
#undef GATHER_INDEX_TYPE_SEQ
#undef GATHER_DATA_TYPE_SEQ
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, GatherFactory, GatherFactoryImpl);
}  // namespace
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
