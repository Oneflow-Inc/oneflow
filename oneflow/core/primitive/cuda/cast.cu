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
#include "oneflow/core/primitive/cast.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow {

namespace primitive {

namespace {

template<typename To, typename From, typename = void>
struct CastFunctor {
  __device__ To operator()(From from) const { return static_cast<To>(from); }
};

template<typename To>
struct CastFunctor<To, half, typename std::enable_if<!std::is_same<To, half>::value>::type> {
  __device__ To operator()(half from) const { return static_cast<To>(static_cast<float>(from)); }
};

template<typename From>
struct CastFunctor<half, From, typename std::enable_if<!std::is_same<From, half>::value>::type> {
  __device__ half operator()(From from) const {
    return static_cast<half>(static_cast<float>(from));
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
void LaunchCast(cudaStream_t stream, const void* from, void* to, size_t count) {
  OF_CUDA_CHECK((cuda::elementwise::Unary<CastFunctor<To, From>, To, From>(
      CastFunctor<To, From>(), count, reinterpret_cast<To*>(to),
      reinterpret_cast<const From*>(from), stream)));
}

using LaunchFn = std::function<void(cudaStream_t /*stream*/, const void* /*from*/, void* /*to*/,
                                    size_t /*count*/)>;

class CastImpl : public Cast {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastImpl);
  explicit CastImpl(LaunchFn launch_fn) : launch_fn_(std::move(launch_fn)) {}
  ~CastImpl() = default;

  void Launch(StreamContext* stream_ctx, const void* from, void* to, size_t count) override {
    auto* cuda_stream_ctx = CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx));
    launch_fn_(cuda_stream_ctx->cuda_stream(), from, to, count);
  }

 private:
  LaunchFn launch_fn_;
};

template<typename From, typename To>
std::unique_ptr<Cast> NewCast() {
  return std::unique_ptr<Cast>(new CastImpl(LaunchCast<From, To>));
}

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
            MAKE_NEW_CAST_ENTRY, CUDA_PRIMITIVE_ALL_TYPE_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ)};

#undef MAKE_NEW_CAST_ENTRY

    const auto it = new_cast_handle.find(std::make_pair(from, to));
    if (it != new_cast_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, CastFactory, CastFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
