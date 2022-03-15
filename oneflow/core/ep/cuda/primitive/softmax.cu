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
#include "oneflow/core/ep/include/primitive/softmax.h"
#include "oneflow/core/ep/include/primitive/log_softmax.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

enum class Algorithm {
  kSoftmax,
  kLogSoftmax,
};

template<Algorithm algorithm, typename T>
void SoftmaxGpu(cudaStream_t cuda_stream, size_t rows, size_t cols, const T* x, T* y) {
  using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
  oneflow::cuda::softmax::DirectLoad<T, ComputeType> load(x, cols);
  oneflow::cuda::softmax::DirectStore<ComputeType, T> store(y, cols);
  if (algorithm == Algorithm::kSoftmax) {
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
        cuda_stream, load, store, rows, cols)));
  } else if (algorithm == Algorithm::kLogSoftmax) {
    OF_CUDA_CHECK((cuda::softmax::DispatchLogSoftmax<decltype(load), decltype(store), ComputeType>(
        cuda_stream, load, store, rows, cols)));
  } else {
    UNIMPLEMENTED();
  }
}

template<typename SoftmaxBase, Algorithm algorithm, typename T>
class SoftmaxImpl : public SoftmaxBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxImpl);
  SoftmaxImpl() = default;
  ~SoftmaxImpl() override = default;

  void Launch(Stream* stream, size_t rows, size_t cols, const void* x, void* y) override {
    cudaStream_t cuda_stream = stream->As<CudaStream>()->cuda_stream();
    SoftmaxGpu<algorithm, T>(cuda_stream, rows, cols, reinterpret_cast<const T*>(x),
                             reinterpret_cast<T*>(y));
  }
};

template<typename SoftmaxBase, Algorithm algorithm, typename T>
std::unique_ptr<SoftmaxBase> NewSoftmax() {
  return std::unique_ptr<SoftmaxBase>(new SoftmaxImpl<SoftmaxBase, algorithm, T>());
}

template<typename FactoryBase, typename SoftmaxBase, Algorithm algorithm>
class GenericSoftmaxFactoryImpl : public FactoryBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GenericSoftmaxFactoryImpl);
  GenericSoftmaxFactoryImpl() = default;
  ~GenericSoftmaxFactoryImpl() override = default;

  std::unique_ptr<SoftmaxBase> New(DataType data_type) override {
#define MAKE_NEW_SOFTMAX_ENTRY(type_cpp, type_proto) \
  {type_proto, NewSoftmax<SoftmaxBase, algorithm, type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<SoftmaxBase>()>>
        new_softmax_handle{
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

using SoftmaxFactoryImpl = GenericSoftmaxFactoryImpl<SoftmaxFactory, Softmax, Algorithm::kSoftmax>;
using LogSoftmaxFactoryImpl =
    GenericSoftmaxFactoryImpl<LogSoftmaxFactory, LogSoftmax, Algorithm::kLogSoftmax>;
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, SoftmaxFactory, SoftmaxFactoryImpl);
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, LogSoftmaxFactory, LogSoftmaxFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
