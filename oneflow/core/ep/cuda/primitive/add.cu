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
#include "oneflow/core/ep/include/primitive/add.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<typename... Args>
struct AddFunctor;

template<typename T>
struct AddFunctor<T> {
  __device__ T operator()(T x) const { return x; }
};

template<typename T, typename U, typename... Args>
struct AddFunctor<T, U, Args...> {
  __device__ T operator()(T x0, U x1, Args... xs) const {
    return x0 + AddFunctor<U, Args...>()(x1, xs...);
  }
};

template<typename T, typename... Args>
__global__ void AddGpu(const Args*... srcs, T* dst, size_t count) {
  CUDA_1D_KERNEL_LOOP_T(size_t, i, count) { dst[i] = AddFunctor<Args...>()(srcs[i]...); }
}

template<typename T, typename... Args>
void LaunchAddGpu(cudaStream_t stream, const Args*... srcs, T* dst, size_t count) {
  AddGpu<T, Args...>
      <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0, stream>>>(srcs..., dst, count);
}

template<typename T>
void DispatchLaunch(cudaStream_t stream, const T* const* srcs, size_t arity, T* dst, size_t count) {
  if (arity == 0) {
    OF_CUDA_CHECK(cudaMemsetAsync(dst, 0, count * sizeof(T), stream));
  } else if (arity == 1) {
    OF_CUDA_CHECK(cudaMemcpyAsync(dst, srcs[0], count * sizeof(T), cudaMemcpyDefault, stream));
  } else if (arity == 2) {
    OF_CUDA_CHECK((cuda::elementwise::Binary<AddFunctor<T, T>, T, T, T>(
        AddFunctor<T, T>(), count, dst, srcs[0], srcs[1], stream)));
  } else if (arity == 3) {
    OF_CUDA_CHECK((cuda::elementwise::Ternary<AddFunctor<T, T, T>, T, T, T, T>(
        AddFunctor<T, T, T>(), count, dst, srcs[0], srcs[1], srcs[2], stream)));
  } else if (arity == 4) {
    LaunchAddGpu<T, T, T, T, T>(stream, srcs[0], srcs[1], srcs[2], srcs[3], dst, count);
  } else if (arity == 5) {
    LaunchAddGpu<T, T, T, T, T, T>(stream, srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], dst, count);
  } else if (arity == 6) {
    LaunchAddGpu<T, T, T, T, T, T, T>(stream, srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5],
                                      dst, count);
  } else if (arity == 7) {
    LaunchAddGpu<T, T, T, T, T, T, T, T>(stream, srcs[0], srcs[1], srcs[2], srcs[3], srcs[4],
                                         srcs[5], srcs[6], dst, count);
  } else if (arity == 8) {
    LaunchAddGpu<T, T, T, T, T, T, T, T, T>(stream, srcs[0], srcs[1], srcs[2], srcs[3], srcs[4],
                                            srcs[5], srcs[6], srcs[7], dst, count);
  } else {
    DispatchLaunch(stream, srcs + 7, arity - 7, dst, count);
    LaunchAddGpu<T, T, T, T, T, T, T, T, T>(stream, srcs[0], srcs[1], srcs[2], srcs[3], srcs[4],
                                            srcs[5], srcs[6], dst, dst, count);
  }
}

template<typename T>
class AddImpl : public Add {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddImpl);
  AddImpl() = default;
  ~AddImpl() override = default;

  using Add::Launch;
  void Launch(Stream* stream, const void* const* srcs, size_t arity, void* dst,
              size_t count) override {
    cudaStream_t cuda_stream = stream->As<CudaStream>()->cuda_stream();
    DispatchLaunch(cuda_stream, reinterpret_cast<const T* const*>(srcs), arity,
                   reinterpret_cast<T*>(dst), count);
  }
};

template<typename T>
std::unique_ptr<Add> NewAdd() {
  return std::unique_ptr<Add>(new AddImpl<T>());
}

class AddFactoryImpl : public AddFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddFactoryImpl);
  AddFactoryImpl() = default;
  ~AddFactoryImpl() override = default;

  std::unique_ptr<Add> New(DataType data_type) override {
#define MAKE_NEW_ADD_ENTRY(type_cpp, type_proto) {type_proto, NewAdd<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<Add>()>> new_add_handle{
        OF_PP_FOR_EACH_TUPLE(MAKE_NEW_ADD_ENTRY, CUDA_PRIMITIVE_ALL_TYPE_SEQ)};

#undef MAKE_NEW_ADD_ENTRY

    const auto it = new_add_handle.find(data_type);
    if (it != new_add_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, AddFactory, AddFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
