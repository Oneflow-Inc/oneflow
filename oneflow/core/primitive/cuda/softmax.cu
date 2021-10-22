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
#include "oneflow/core/primitive/include/softmax.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow {

namespace primitive {

namespace {

template<typename T>
class SoftmaxImpl : public Softmax {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxImpl);
  SoftmaxImpl() = default;
  ~SoftmaxImpl() override = default;

  using Softmax::Launch;
  void Launch(StreamContext* stream_ctx, size_t rows, size_t cols, const void* x,
              void* y) override {
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    oneflow::cuda::softmax::DirectLoad<T, ComputeType> load(reinterpret_cast<const T*>(x), cols);
    oneflow::cuda::softmax::DirectStore<ComputeType, T> store(reinterpret_cast<T*>(y), cols);
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
        cuda_stream, load, store, rows, cols)));
  }
};

template<typename T>
std::unique_ptr<Softmax> NewSoftmax() {
  return std::unique_ptr<Softmax>(new SoftmaxImpl<T>());
}

class SoftmaxFactoryImpl : public SoftmaxFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxFactoryImpl);
  SoftmaxFactoryImpl() = default;
  ~SoftmaxFactoryImpl() override = default;

  std::unique_ptr<Softmax> New(DataType data_type) override {
#define MAKE_NEW_SOFTMAX_ENTRY(type_cpp, type_proto) {type_proto, NewSoftmax<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<Softmax>()>> new_softmax_handle{
        OF_PP_FOR_EACH_TUPLE(MAKE_NEW_SOFTMAX_ENTRY, CUDA_PRIMITIVE_FLOATING_TYPE_SEQ)};

#undef MAKE_NEW_SOFTMAX_ENTRY

    const auto it = new_softmax_handle.find(data_type);
    if (it != new_softmax_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, SoftmaxFactory, SoftmaxFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
