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
#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/gather.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace gather {

namespace {

template<typename ParamsType, typename IndicesType>
void GatherCpu(size_t batch_dim_size, size_t outer_dim_size, size_t gather_dim_size,
               size_t inner_dim_size, size_t offset, const ParamsType* in, size_t indices_size,
               const IndicesType* indices, ParamsType* out) {
  size_t out_size = outer_dim_size * indices_size * inner_dim_size;
  NdIndexOffsetHelper<IndicesType, 4> in_helper(batch_dim_size, outer_dim_size, gather_dim_size,
                                                inner_dim_size);
  NdIndexOffsetHelper<IndicesType, 4> out_helper(batch_dim_size, outer_dim_size,
                                                 indices_size / batch_dim_size, inner_dim_size);
  NdIndexOffsetHelper<IndicesType, 2> indices_helper(batch_dim_size, indices_size / batch_dim_size);
  IndicesType index[4];
  IndicesType indices_index[2];
  for (size_t i = 0; i < out_size; i++) {
    out_helper.OffsetToNdIndex(i, index);
    indices_index[0] = index[0];  // batch_dim_index
    indices_index[1] = index[2];  // gather_dim_index
    index[2] = indices[indices_helper.NdIndexToOffset(indices_index)] - offset;
    ParamsType result{};
    if (index[2] >= 0 && index[2] <= gather_dim_size) {
      result = in[in_helper.NdIndexToOffset(index)];
    }
    out[i] = result;
  }
}

template<typename ParamsType, typename IndicesType>
class GatherImpl : public Gather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() override = default;

  using Gather::Launch;
  void Launch(Stream* stream, size_t batch_dim_size, size_t outer_dim_size, size_t gather_dim_size,
              size_t inner_dim_size, size_t offset, const void* in, size_t indices_size,
              const void* indices, void* out) override {
    GatherCpu<ParamsType, IndicesType>(
        batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size, offset,
        reinterpret_cast<const ParamsType*>(in), indices_size,
        reinterpret_cast<const IndicesType*>(indices), reinterpret_cast<ParamsType*>(out));
  }
};

template<typename ParamsType, typename IndicesType>
std::unique_ptr<Gather> NewGather() {
  return std::unique_ptr<Gather>(new GatherImpl<ParamsType, IndicesType>());
}

class GatherFactoryImpl : public GatherFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactoryImpl);
  GatherFactoryImpl() = default;
  ~GatherFactoryImpl() override = default;

  std::unique_ptr<Gather> New(DataType params_type, DataType indices_type) override {
#define MAKE_NEW_GATHER_ENTRY(params_type_pair, indices_type_pair)                            \
  {std::make_pair(OF_PP_PAIR_SECOND(params_type_pair), OF_PP_PAIR_SECOND(indices_type_pair)), \
   NewGather<OF_PP_PAIR_FIRST(params_type_pair), OF_PP_PAIR_FIRST(indices_type_pair)>},

    static const std::map<std::pair<DataType, DataType>, std::function<std::unique_ptr<Gather>()>>
        new_gather_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_NEW_GATHER_ENTRY, CUDA_PRIMITIVE_ALL_TYPE_SEQ, CUDA_PRIMITIVE_INT_TYPE_SEQ)};

#undef MAKE_NEW_GATHER_ENTRY

    const auto it = new_gather_handle.find(std::make_pair(params_type, indices_type));
    if (it != new_gather_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, GatherFactory, GatherFactoryImpl);

}  // namespace

}  // namespace gather

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
