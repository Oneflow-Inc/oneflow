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
#include "oneflow/core/ep/include/primitive/tensor_fill.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<size_t size>
using Storage = typename std::aligned_storage<size, size>::type;

template<typename T, size_t pack>
union Pack {
  static constexpr size_t size = sizeof(T) * pack;
  explicit __device__ __host__ Pack(const T value) {
    static_assert(sizeof(Pack) == size, "");
    static_assert(alignof(Pack) == size, "");
#pragma unroll
    for (size_t i = 0; i < pack; ++i) { elem[i] = value; }
  }
  T elem[pack];
  Storage<size> storage;
};

template<typename T, size_t pack>
__global__ void TensorFillGpu(T* dst, const T* value, size_t count) {
  const size_t pack_count = count / pack;
  const T fill_value = value[0];
  Pack<T, pack> pack_value(fill_value);
  auto* pack_dst = reinterpret_cast<decltype(pack_value.storage)*>(dst);
  CUDA_1D_KERNEL_LOOP_T(size_t, i, pack_count) { pack_dst[i] = pack_value.storage; }
  T* tail_dst = dst + pack_count * pack;
  const size_t tail_count = count - pack_count * pack;
  CUDA_1D_KERNEL_LOOP_T(size_t, i, tail_count) { tail_dst[i] = fill_value; }
}

template<typename T, size_t pack>
typename std::enable_if<(pack != 0), void>::type LaunchPackTensorFill(cudaStream_t stream, T* dst,
                                                                      const T* value,
                                                                      size_t count) {
  TensorFillGpu<T, pack>
      <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0, stream>>>(dst, value, count);
}

template<typename T, size_t pack>
typename std::enable_if<(pack == 0), void>::type LaunchPackTensorFill(cudaStream_t stream, T* dst,
                                                                      const T* value,
                                                                      size_t count) {
  LOG(FATAL) << "wrong alignment";
}

template<typename T>
void LaunchTensorFill(cudaStream_t stream, T* dst, const T* value, size_t count) {
  auto uintptr = reinterpret_cast<std::uintptr_t>(dst);
  if (uintptr % 16 == 0) {
    LaunchPackTensorFill<T, 16 / sizeof(T)>(stream, dst, value, count);
  } else if (uintptr % 8 == 0) {
    LaunchPackTensorFill<T, 8 / sizeof(T)>(stream, dst, value, count);
  } else if (uintptr % 4 == 0) {
    LaunchPackTensorFill<T, 4 / sizeof(T)>(stream, dst, value, count);
  } else if (uintptr % 2 == 0) {
    LaunchPackTensorFill<T, 2 / sizeof(T)>(stream, dst, value, count);
  } else {
    LaunchPackTensorFill<T, 1 / sizeof(T)>(stream, dst, value, count);
  }
}

template<typename T>
class TensorFillImpl : public TensorFill {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorFillImpl);
  TensorFillImpl() = default;
  ~TensorFillImpl() override = default;

  void Launch(Stream* stream, const void* src, void* dst, size_t count) override {
    cudaStream_t cuda_stream = stream->As<CudaStream>()->cuda_stream();
    const T* value = reinterpret_cast<const T*>(src);
    LaunchTensorFill<T>(cuda_stream, reinterpret_cast<T*>(dst), value, count);
  }
};

template<typename T>
std::unique_ptr<TensorFill> NewTensorFill() {
  return std::unique_ptr<TensorFill>(new TensorFillImpl<T>());
}

class TensorFillFactoryImpl : public TensorFillFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorFillFactoryImpl);
  TensorFillFactoryImpl() = default;
  ~TensorFillFactoryImpl() override = default;

  std::unique_ptr<TensorFill> New(DataType data_type) override {
#define MAKE_NEW_TENSOR_FILL_ENTRY(type_cpp, type_proto) {type_proto, NewTensorFill<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<TensorFill>()>> new_fill_handle{
        OF_PP_FOR_EACH_TUPLE(MAKE_NEW_TENSOR_FILL_ENTRY, CUDA_PRIMITIVE_ALL_TYPE_SEQ)};

#undef MAKE_NEW_TENSOR_FILL_ENTRY

    const auto it = new_fill_handle.find(data_type);
    if (it != new_fill_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, TensorFillFactory, TensorFillFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
