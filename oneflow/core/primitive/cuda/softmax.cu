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
#include "oneflow/core/primitive/include/log_softmax.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow {

namespace primitive {

namespace {

enum class Algorithm {
  kSoftmax,
  kLogSoftmax,
};

template<typename T, Algorithm algorithm>
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

template<typename T>
class SoftmaxImpl : public Softmax {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxImpl);
  SoftmaxImpl() = default;
  ~SoftmaxImpl() override = default;

  void Launch(StreamContext* stream_ctx, size_t rows, size_t cols, const void* x,
              void* y) override {
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    SoftmaxGpu<T, Algorithm::kSoftmax>(cuda_stream, rows, cols, reinterpret_cast<const T*>(x),
                                       reinterpret_cast<T*>(y));
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

template<typename T>
class LogSoftmaxImpl : public LogSoftmax {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogSoftmaxImpl);
  LogSoftmaxImpl() = default;
  ~LogSoftmaxImpl() override = default;

  void Launch(StreamContext* stream_ctx, size_t rows, size_t cols, const void* x,
              void* y) override {
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    SoftmaxGpu<T, Algorithm::kLogSoftmax>(cuda_stream, rows, cols, reinterpret_cast<const T*>(x),
                                          reinterpret_cast<T*>(y));
  }
};

template<typename T>
std::unique_ptr<LogSoftmax> NewLogSoftmax() {
  return std::unique_ptr<LogSoftmax>(new LogSoftmaxImpl<T>());
}

class LogSoftmaxFactoryImpl : public LogSoftmaxFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogSoftmaxFactoryImpl);
  LogSoftmaxFactoryImpl() = default;
  ~LogSoftmaxFactoryImpl() override = default;

  std::unique_ptr<LogSoftmax> New(DataType data_type) override {
#define MAKE_NEW_LOG_SOFTMAX_ENTRY(type_cpp, type_proto) {type_proto, NewLogSoftmax<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<LogSoftmax>()>>
        new_log_softmax_handle{
            OF_PP_FOR_EACH_TUPLE(MAKE_NEW_LOG_SOFTMAX_ENTRY, CUDA_PRIMITIVE_FLOATING_TYPE_SEQ)};

#undef MAKE_NEW_LOG_SOFTMAX_ENTRY

    const auto it = new_log_softmax_handle.find(data_type);
    if (it != new_log_softmax_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, LogSoftmaxFactory, LogSoftmaxFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
