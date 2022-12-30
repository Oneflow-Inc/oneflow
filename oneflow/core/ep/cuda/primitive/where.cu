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
#include "oneflow/core/ep/cuda/primitive/where.cuh"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace where_cuda_details {

template<typename T, typename CondT>
class WhereImpl : public Where {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereImpl);
  explicit WhereImpl() = default;
  ~WhereImpl() override = default;

  void Launch(Stream* stream, size_t cond_ndim, const int64_t* cond_dims, const void* cond,
              size_t x_ndim, const int64_t* x_dims, const void* x, size_t y_ndim,
              const int64_t* y_dims, const void* y, void* z) override {
    auto* cuda_stream = stream->As<CudaStream>();
    OF_CUDA_CHECK(where_cuda_details::Launch(
        cuda_stream->cuda_stream(), cond_ndim, cond_dims, reinterpret_cast<const CondT*>(cond),
        x_ndim, x_dims, reinterpret_cast<const T*>(x), y_ndim, y_dims,
        reinterpret_cast<const T*>(y), reinterpret_cast<T*>(z)));
  }
};

template<typename T, typename CondT>
std::unique_ptr<Where> NewWhere() {
  return std::unique_ptr<Where>(new WhereImpl<T, CondT>());
}

#define CUDA_PRIMITIVE_WHERE_DATA_TYPE_SEQ \
  CUDA_PRIMITIVE_BOOL_TYPE_SEQ             \
  CUDA_PRIMITIVE_INT32_TYPE_SEQ            \
  CUDA_PRIMITIVE_FLOAT_TYPE_SEQ            \
  CUDA_PRIMITIVE_DOUBLE_TYPE_SEQ           \
  CUDA_PRIMITIVE_FLOAT16_TYPE_SEQ          \
  CUDA_PRIMITIVE_BFLOAT16_TYPE_SEQ

#define CUDA_PRIMITIVE_WHERE_COND_TYPE_SEQ \
  CUDA_PRIMITIVE_BOOL_TYPE_SEQ             \
  CUDA_PRIMITIVE_INT32_TYPE_SEQ

class WhereFactoryImpl : public WhereFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereFactoryImpl);
  WhereFactoryImpl() = default;
  ~WhereFactoryImpl() override = default;

#define MAKE_WHERE_CREATOR_ENTRY(dtype_pair, ctype_pair)                         \
  {std::make_pair(OF_PP_PAIR_SECOND(dtype_pair), OF_PP_PAIR_SECOND(ctype_pair)), \
   NewWhere<OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ctype_pair)>},

  std::unique_ptr<Where> New(DataType data_type, DataType cond_type) override {
    static const std::map<std::pair<DataType, DataType>, std::function<std::unique_ptr<Where>()>>
        where_creator_regsitry{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_WHERE_CREATOR_ENTRY, CUDA_PRIMITIVE_WHERE_DATA_TYPE_SEQ,
            CUDA_PRIMITIVE_WHERE_COND_TYPE_SEQ)};

    auto it = where_creator_regsitry.find(std::make_pair(data_type, cond_type));
    if (it != where_creator_regsitry.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
#undef MAKE_WHERE_CREATOR_ENTRY
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, WhereFactory, WhereFactoryImpl);

}  // namespace where_cuda_details

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
