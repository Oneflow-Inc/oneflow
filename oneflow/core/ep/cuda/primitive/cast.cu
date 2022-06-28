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
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<typename To, typename From, typename = void>
struct CastFunctor {
  __device__ To operator()(From from) const { return static_cast<To>(from); }
};

template<typename To>
struct CastFunctor<To, half, typename std::enable_if<!std::is_same<To, half>::value>::type> {
  __device__ To operator()(half from) const { return static_cast<To>(static_cast<float>(from)); }

  __device__ void Apply2(To* to, const half* from) const {
    const float2 f2 = __half22float2(*reinterpret_cast<const half2*>(from));
    to[0] = static_cast<To>(f2.x);
    to[1] = static_cast<To>(f2.y);
  }
};

template<typename From>
struct CastFunctor<half, From, typename std::enable_if<!std::is_same<From, half>::value>::type> {
  __device__ half operator()(From from) const {
    return static_cast<half>(static_cast<float>(from));
  }

  __device__ void Apply2(half* to, const From* from) const {
    float2 f2;
    f2.x = static_cast<float>(from[0]);
    f2.y = static_cast<float>(from[1]);
    *reinterpret_cast<half2*>(to) = __float22half2_rn(f2);
  }
};

#if CUDA_VERSION >= 11000

template<typename To>
struct CastFunctor<To, nv_bfloat16,
                   typename std::enable_if<!(std::is_same<To, nv_bfloat16>::value
                                             || std::is_same<To, half>::value)>::type> {
  __device__ To operator()(nv_bfloat16 from) const {
    return static_cast<To>(static_cast<float>(from));
  }
};

template<typename From>
struct CastFunctor<nv_bfloat16, From,
                   typename std::enable_if<!(std::is_same<From, nv_bfloat16>::value
                                             || std::is_same<From, half>::value)>::type> {
  __device__ nv_bfloat16 operator()(From from) const {
    return static_cast<nv_bfloat16>(static_cast<float>(from));
  }
};

#endif  // CUDA_VERSION >= 11000

template<typename From, typename To>
class CastImpl : public Cast {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastImpl);
  explicit CastImpl() = default;
  ~CastImpl() override = default;

  void Launch(Stream* stream, const void* from, void* to, size_t count) override {
    auto* cuda_stream = stream->As<CudaStream>();
    OF_CUDA_CHECK((cuda::elementwise::Unary<CastFunctor<To, From>, To, From>(
        CastFunctor<To, From>(), count, reinterpret_cast<To*>(to),
        reinterpret_cast<const From*>(from), cuda_stream->cuda_stream())));
  }
};

template<typename From, typename To>
std::unique_ptr<Cast> NewCast() {
  return std::unique_ptr<Cast>(new CastImpl<From, To>());
}

#define CUDA_PRIMITIVE_CAST_TYPE_SEQ \
  CUDA_PRIMITIVE_BOOL_TYPE_SEQ       \
  CUDA_PRIMITIVE_CHAR_TYPE_SEQ       \
  CUDA_PRIMITIVE_INT8_TYPE_SEQ       \
  CUDA_PRIMITIVE_UINT8_TYPE_SEQ      \
  CUDA_PRIMITIVE_INT32_TYPE_SEQ      \
  CUDA_PRIMITIVE_UINT32_TYPE_SEQ     \
  CUDA_PRIMITIVE_INT64_TYPE_SEQ      \
  CUDA_PRIMITIVE_UINT64_TYPE_SEQ     \
  CUDA_PRIMITIVE_FLOAT_TYPE_SEQ      \
  CUDA_PRIMITIVE_DOUBLE_TYPE_SEQ     \
  CUDA_PRIMITIVE_FLOAT16_TYPE_SEQ    \
  CUDA_PRIMITIVE_BFLOAT16_TYPE_SEQ

class CastFactoryImpl : public CastFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastFactoryImpl);
  CastFactoryImpl() = default;
  ~CastFactoryImpl() override = default;

  std::unique_ptr<Cast> New(DataType from, DataType to) override {
#define MAKE_NEW_CAST_ENTRY(from_pair, to_pair)                              \
  {std::make_pair(OF_PP_PAIR_SECOND(from_pair), OF_PP_PAIR_SECOND(to_pair)), \
   NewCast<OF_PP_PAIR_FIRST(from_pair), OF_PP_PAIR_FIRST(to_pair)>},

    static const std::map<std::pair<DataType, DataType>, std::function<std::unique_ptr<Cast>()>>
        new_cast_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_NEW_CAST_ENTRY, CUDA_PRIMITIVE_CAST_TYPE_SEQ, CUDA_PRIMITIVE_CAST_TYPE_SEQ)};

#undef MAKE_NEW_CAST_ENTRY

    const auto it = new_cast_handle.find(std::make_pair(from, to));
    if (it != new_cast_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, CastFactory, CastFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
